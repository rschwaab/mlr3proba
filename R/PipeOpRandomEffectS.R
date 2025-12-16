# --------------- PipeOp: Random effects from a Long table -----------------
PipeOpRandomEffectLong <- R6::R6Class(
  "PipeOpRandomEffectLong",
  inherit = mlr3pipelines::PipeOp,

  public = list(
    initialize = function(id = "proba.random_effect_long") {
      super$initialize(
        id = id,
        param_set = paradox::ps(
          long_id_col    = paradox::p_uty(init = "id",      tags = c("train","predict")),
          long_time_col  = paradox::p_uty(init = "fuptime", tags = c("train","predict")),
          feature_cols   = paradox::p_uty(tags = c("train","predict"))
        ),
        # Two inputs: TaskSurv + Long
        input = data.table::data.table(
          name   = c("task","long"),
          train  = c("Task","Long"),
          predict= c("Task","Long")
        ),
        # One output: TaskSurv augmented with RE features
        output = data.table::data.table(
          name   = "output",
          train  = "Task",
          predict= "Task"
        ),
        packages = c("mlr3","mlr3proba","data.table","lme4","checkmate")
      )
    }
  ),

  private = list(

    .train = function(inputs) {
      task <- inputs[[1L]]
      long <- inputs[[2L]]        # inherits "Long" (is a data.table under the hood)
      long <- data.table::as.data.table(long)

      pv <- self$param_set$values
      checkmate::assert_string(pv$long_id_col)
      checkmate::assert_string(pv$long_time_col)

      id_col    <- private$.get_id_col(task)
      feat_cols <- private$.get_feature_cols(long, pv$long_id_col, pv$long_time_col, pv$feature_cols)

      keep_cols <- unique(c(id_col, task$feature_names, task$target_names))
      dt_keep   <- data.table::as.data.table(task$data(cols = keep_cols))
      id_orig   <- dt_keep[[id_col]]
      dt_keep[, (id_col) := as.character(get(id_col))]
      task_ids  <- unique(dt_keep[[id_col]])

      dt_re <- unique(dt_keep[, ..id_col])
      params <- setNames(vector("list", length(feat_cols)), feat_cols)

      for (nm in feat_cols) {
        tab <- private$.mk_tab(long, pv$long_id_col, pv$long_time_col, nm)
        private$.assert_all_task_ids_have_data(task_ids, tab, nm)

        mod <- lme4::lmer(value ~ arg + (1 + arg | id), data = tab)

        feats <- private$.ranefs_to_dt(mod, id_col, nm)
        dt_re <- feats[dt_re, on = id_col]

        params[[nm]] <- private$.extract_lmm_params(mod)
      }

      new_task <- private$.finalize_task(task, dt_re, dt_keep, id_col, id_orig)

      self$state <- list(
        params        = params,
        feature_cols  = feat_cols,
        id_col        = id_col,
        long_id_col   = pv$long_id_col,
        long_time_col = pv$long_time_col
      )
      list(new_task)
    },

    .predict = function(inputs) {
      task <- inputs[[1L]]
      long <- inputs[[2L]]
      long <- data.table::as.data.table(long)
      st   <- self$state
      if (is.null(st$params)) stop("PipeOpRandomEffectLong was not trained.")

      id_col    <- st$id_col
      feat_cols <- st$feature_cols

      keep_cols <- unique(c(id_col, task$feature_names, task$target_names))
      dt_keep   <- data.table::as.data.table(task$data(cols = keep_cols))
      id_orig   <- dt_keep[[id_col]]
      dt_keep[, (id_col) := as.character(get(id_col))]
      task_ids  <- unique(dt_keep[[id_col]])

      dt_re <- unique(dt_keep[, ..id_col])

      for (nm in feat_cols) {
        tab <- private$.mk_tab(long, st$long_id_col, st$long_time_col, nm)
        private$.assert_all_task_ids_have_data(task_ids, tab, nm)

        feats <- private$.prc_predict_feats(st$params[[nm]], tab, id_col)

        re_cols <- sprintf("%s_%s", nm, c("random_intercept", "random_slope"))
        data.table::setnames(
          feats,
          old = c("random_intercept", "random_slope"),
          new = re_cols,
          skip_absent = FALSE
        )

        dt_re <- feats[dt_re, on = id_col]
      }

      new_task <- private$.finalize_task(task, dt_re, dt_keep, id_col, id_orig)
      list(new_task)
    },

    # ---- helpers ----
    .mk_tab = function(long, id_col, time_col, feature_name) {
      if (!all(c(id_col, time_col, feature_name) %in% names(long)))
        stop(sprintf("Long table is missing required columns for '%s'.", feature_name))
      tab <- data.table::data.table(
        id    = as.character(long[[id_col]]),
        arg   = as.numeric(long[[time_col]]),
        value = as.numeric(long[[feature_name]])
      )
      tab <- tab[!is.na(id) & !is.na(arg) & !is.na(value)]
      if (nrow(tab) == 0L) stop(sprintf("After dropping NAs, no rows remain for '%s'.", feature_name))
      tab
    },

    .assert_all_task_ids_have_data = function(task_ids, tab, nm) {
      missing <- setdiff(task_ids, unique(tab$id))
      if (length(missing))
        stop(sprintf("Feature '%s': missing longitudinal data for %d id(s), e.g. %s",
                     nm, length(missing), paste(head(missing, 5), collapse = ", ")))
      invisible(TRUE)
    },


    .ranefs_to_dt = function(mod, id_col, nm) {
      re <- lme4::ranef(mod)$id
      dt <- data.table::as.data.table(re, keep.rownames = id_col)
      data.table::set(dt, j = id_col, value = as.character(dt[[id_col]]))
      re_cols <- sprintf("%s_%s", nm, c("random_intercept","random_slope"))
      data.table::setnames(dt, old = c("(Intercept)","arg"), new = re_cols)
      dt
    },

    .extract_lmm_params = function(mod) {
      list(
        beta   = as.numeric(lme4::fixef(mod)),
        D      = as.matrix(lme4::VarCorr(mod)$id),
        sigma2 = lme4::getME(mod, "sigma")^2
      )
    },

    .prc_predict_feats = function(params, tab, id_col) {
      beta   <- params$beta
      D      <- params$D
      sigma2 <- params$sigma2
      ids <- unique(tab$id)

      u_hat <- matrix(NA_real_, nrow = length(ids), ncol = 2L,
                      dimnames = list(ids, c("random_intercept","random_slope")))
      for (j in seq_along(ids)) {
        dj <- tab[tab$id == ids[j]]
        X  <- cbind(1, dj$arg)
        Z  <- X
        r  <- dj$value - as.numeric(X %*% beta)
        V  <- Z %*% D %*% t(Z) + sigma2 * diag(nrow(Z))
        u_hat[j,] <- as.numeric(D %*% t(Z) %*% solve(V, r))
      }
      feats <- data.table::as.data.table(u_hat)
      feats[, (id_col) := rownames(u_hat)]
      feats[]
    },

    .finalize_task = function(task, dt_re, dt_keep, id_col, id_orig) {
      dt_new <- dt_re[dt_keep, on = id_col]
      stopifnot(nrow(dt_new) == nrow(dt_keep))
      if (is.factor(id_orig)) {
        dt_new[, (id_col) := factor(get(id_col), levels = levels(id_orig))]
      }
      backend <- mlr3::as_data_backend(dt_new)
      new_task <- mlr3proba::TaskSurv$new(
        id      = task$id,
        backend = backend,
        time    = task$target_names[1L],
        event   = task$target_names[2L]
      )
      new_task$col_roles$group   <- id_col
      new_task$col_roles$feature <- setdiff(new_task$col_roles$feature, id_col)
      new_task
    },

    .get_id_col = function(task) {
      id_col <- task$col_roles$group
      if (length(id_col) == 0L) stop("No group column set on task.")
      if (length(id_col) != 1L) stop("Need exactly one group column.")
      id_col[[1L]]
    },

    .get_feature_cols = function(long, id_col, time_col, feature_cols) {
      if (!is.null(feature_cols)) {
        checkmate::assert_character(feature_cols, min.chars = 1)
        return(feature_cols)
      }
      num <- names(long)[vapply(long, is.numeric, FUN.VALUE = logical(1))]
      num <- setdiff(num, c(id_col, time_col))
      if (!length(num)) stop("Could not auto-detect numeric feature columns in 'long'. Set 'feature_cols'.")
      num
    }
  )
)

register_pipeop("random_effect_long", PipeOpRandomEffectLong)

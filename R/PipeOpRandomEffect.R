PipeOpRandomEffect <- R6::R6Class(
  "PipeOpRandomEffect",
  inherit = mlr3pipelines::PipeOp,
  public = list(
    initialize = function(id = "proba.random_effect") {
      super$initialize(
        id = id,
        input  = data.table::data.table(name = "input",  train = "Task", predict = "Task"),
        output = data.table::data.table(name = "output", train = "Task", predict = "Task"),
        packages = c("mlr3", "mlr3proba", "lme4", "data.table")
      )
    }
  ),
  private = list(
    .train = function(inputs) {
      task <- inputs[[1L]]
      id_col <- private$.get_id_col(task)

      fun_cols <- private$.tfd_cols(task)
      if (length(fun_cols) == 0L) stop("No functional (tfd_*) columns found.")

      prep <- private$.prep_tables(task, id_col, fun_cols)

      models <- setNames(vector("list", length(fun_cols)), fun_cols)
      dt_re <- prep$dt_re

      for (nm in fun_cols) {
        tab <- private$.tfd_tab(prep$dt_fun, nm)
        private$.check_ids(prep$task_ids, tab, nm)

        models[[nm]] <- lme4::lmer(value ~ arg + (1 + arg | id), data = tab)

        feats <- private$.extract_train_feats(models[[nm]], id_col)
        dt_re <- private$.join_feats(dt_re, feats, id_col, nm, context = "train")
      }

      new_task <- private$.finalize_task(
        task   = task,
        dt_re  = dt_re,
        dt_keep = prep$dt_keep,
        id_col = id_col,
        id_orig = prep$id_orig
      )

      self$state <- list(
        models   = models,
        fun_cols = fun_cols,
        id_col   = id_col
      )

      list(new_task)
    },

    .predict = function(inputs) {
      task <- inputs[[1L]]
      st <- self$state

      fun_cols <- st$fun_cols
      if (length(fun_cols) == 0L) return(list(task))

      id_col <- st$id_col
      prep <- private$.prep_tables(task, id_col, fun_cols)

      dt_re <- prep$dt_re

      for (nm in fun_cols) {
        tab <- private$.tfd_tab(prep$dt_fun, nm)
        private$.check_ids(prep$task_ids, tab, nm)

        feats <- private$.prc_predict_feats(st$models[[nm]], tab, id_col)
        dt_re <- private$.join_feats(dt_re, feats, id_col, nm, context = "predict")
      }

      new_task <- private$.finalize_task(
        task   = task,
        dt_re  = dt_re,
        dt_keep = prep$dt_keep,
        id_col = id_col,
        id_orig = prep$id_orig
      )

      list(new_task)
    },
    .prep_tables = function(task, id_col, fun_cols) {
      dt_fun <- task$data(cols = fun_cols)

      keep_cols <- unique(c(id_col, setdiff(task$feature_names, fun_cols), task$target_names))
      dt_keep <- data.table::as.data.table(task$data(cols = keep_cols))

      id_orig <- dt_keep[[id_col]]
      dt_keep[, (id_col) := as.character(get(id_col))]
      task_ids <- unique(dt_keep[[id_col]])

      # unique ids only -> prevents cartesian explosion if task ever has repeated ids
      dt_re <- unique(dt_keep[, ..id_col])

      list(
        dt_fun   = dt_fun,
        dt_keep  = dt_keep,
        id_orig  = id_orig,
        task_ids = task_ids,
        dt_re    = dt_re
      )
    },
    .tfd_tab = function(dt_fun, nm) {
      x <- dt_fun[[nm]]
      tab <- as.data.frame(x, unnest = TRUE)
      stats::na.omit(tab)
    },
    .check_ids = function(task_ids, tab, nm) {
      tab_ids <- unique(as.character(tab$id))

      missing_ids <- setdiff(task_ids, tab_ids)
      if (length(missing_ids)) {
        stop(sprintf(
          "'%s': no observations after na.omit for %d subject(s), e.g. %s",
          nm, length(missing_ids), paste(head(missing_ids, 5), collapse = ", ")
        ))
      }

      extra_ids <- setdiff(tab_ids, task_ids)
      if (length(extra_ids)) {
        stop(sprintf(
          "'%s': found %d id(s) in tfd not present in task, e.g. %s",
          nm, length(extra_ids), paste(head(extra_ids, 5), collapse = ", ")
        ))
      }

      invisible(TRUE)
    },
    .extract_train_feats = function(mod, id_col) {
      data.table::as.data.table(lme4::ranef(mod)$id, keep.rownames = id_col)
    },
    .prc_predict_feats = function(mod, tab, id_col) {
      beta <- lme4::fixef(mod)
      D <- as.matrix(lme4::VarCorr(mod)$id)
      sigma2 <- lme4::getME(mod, "sigma")^2

      ids <- unique(as.character(tab$id))

      u_hat <- matrix(
        NA_real_, nrow = length(ids), ncol = 2L,
        dimnames = list(ids, c("random_intercept", "random_slope"))
      )

      for (j in seq_along(ids)) {
        id_j <- ids[j]
        dat_j <- tab[as.character(tab$id) == id_j, ]

        y <- dat_j$value
        arg <- dat_j$arg

        X <- cbind(1, arg)
        Z <- X
        residual <- y - X %*% beta
        V <- Z %*% D %*% t(Z) + sigma2 * diag(nrow(Z))

        u_hat[j, ] <- as.numeric(D %*% t(Z) %*% solve(V, residual))
      }

      feats <- data.table::as.data.table(u_hat)
      feats[, (id_col) := rownames(u_hat)]
      feats
    },
    .join_feats = function(dt_re, feats, id_col, nm, context = "") {
      feats[, (id_col) := as.character(get(id_col))]

      re_cols <- sprintf("%s_%s", nm, c("random_intercept", "random_slope"))

      if (all(c("(Intercept)", "arg") %in% names(feats))) {
        data.table::setnames(feats, c(id_col, "(Intercept)", "arg"), c(id_col, re_cols))
      } else {
        data.table::setnames(feats, c("random_intercept", "random_slope"), re_cols)
      }

      dt_re <- feats[dt_re, on = id_col]
      private$.assert_all_ids_present(dt_re, id_col, re_cols, context = paste0(context, "/", nm))
      dt_re
    },
    .finalize_task = function(task, dt_re, dt_keep, id_col, id_orig) {
      dt_new <- dt_re[dt_keep, on = id_col]
      stopifnot(nrow(dt_new) == task$nrow)

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

      new_task$col_roles$group = id_col
      new_task$col_roles$feature = setdiff(new_task$col_roles$feature, id_col)

      new_task
    },
    .tfd_cols = function(task, types = c("tfd_irreg", "tfd_reg")) {
      ft <- task$feature_types
      ft[type %in% types, id]
    },
    .assert_all_ids_present = function(dt, id_col, new_cols, context = "") {
      miss <- dt[Reduce(`|`, lapply(new_cols, function(cc) is.na(get(cc)))), get(id_col)]
      if (length(miss)) {
        stop(sprintf(
          "%sMissing random-effect features for %d id(s), e.g.: %s",
          if (nzchar(context)) paste0(context, ": ") else "",
          length(unique(miss)),
          paste(head(unique(miss), 5), collapse = ", ")
        ))
      }
      invisible(TRUE)
    },
    .get_id_col = function(task) {
      id_col <- task$col_roles$group
      if (length(id_col) == 0L) {
        if ("subject_id" %in% task$col_names) return("subject_id")
        stop("No group column set on task and no 'subject_id' column found.")
      }
      if (length(id_col) != 1L) stop("Need exactly one group column.")
      id_col[[1L]]
    }
  )
)

register_pipeop("random_effect", PipeOpRandomEffect)


# https://chatgpt.com/c/6909faa8-6ad8-8328-b931-673bf45e6766

# - Make sure ids work properly (match on subject ids)
# - Create Version 1: save only relevant stuff during .train in state and use this
# - Create Version 2: save full model and use lme4 predict function instead of manual calculations

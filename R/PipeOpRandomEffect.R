PipeOpRandomEffect <- R6::R6Class(
  "PipeOpRandomEffect",
  inherit = mlr3pipelines::PipeOp,
  public = list(
    initialize = function(id = "fda.random_effect") {
      super$initialize(
        id = id,
        param_set = paradox::ps(
          min_points = paradox::p_int(lower = 1, init = 2)
        ),
        input  = data.table::data.table(name = "input",  train = "Task", predict = "Task"),
        output = data.table::data.table(name = "output", train = "Task", predict = "Task")
      )
    }
  ),
  private = list(

    .train = function(inputs) {
      min_pts <- self$param_set$values$min_points
      task <- inputs[[1L]]
      fun_cols <- private$.tfd_cols(task)
      if (length(fun_cols) == 0) stop("No functional (tfd_*) columns found.")
      dt_fun <- task$data(cols = fun_cols)
      models <- setNames(vector("list", length(fun_cols)), fun_cols)
      feat_pieces <- vector("list", length(fun_cols)); names(feat_pieces) <- fun_cols
      # fit models
      for (nm in fun_cols) {
        x <- dt_fun[[nm]]
        tab <- as.data.frame(x, unnest = TRUE)
        tab <- stats::na.omit(tab)
        # keep only ids with enough unique time points
        # keep_ids <- names(which(tapply(tab$arg, tab$id, function(a) length(unique(a)) >= min_pts)))
        # tab <- tab[tab$id %in% keep_ids, , drop = FALSE]

        # fit model
        models[[nm]] <- lme4::lmer(value ~ arg + (1 + arg | id), data = tab)
        feats <- lme4::ranef(models[[nm]])$id
        setDT(feats)
        setnames(feats, sprintf("%s_%s", nm, c("random_intercept", "random_slope")))
        feat_pieces[[nm]] <- feats
      }
      feat_dt <- do.call(cbind, unname(feat_pieces))
      keep_cols <- c(setdiff(task$feature_names, fun_cols), task$target_names)
      dt_keep   <- task$data(cols = keep_cols)
      dt_new <- data.table::as.data.table(cbind(dt_keep, feat_dt))
      stopifnot(nrow(dt_new) == task$nrow)
      backend <- mlr3::as_data_backend(dt_new)
      new_task <- mlr3proba::TaskSurv$new(
        id = task$id,
        backend = backend,
        time  = task$target_names[1L],
        event = task$target_names[2L]
      )



      self$state <- list(
        models   = models,
        fun_cols = fun_cols,
        re_names = names(feat_dt)
      )

      list(new_task)
    },

    .predict = function(inputs) {
      task <- inputs[[1L]]
      st   <- self$state
    },
    .tfd_cols = function(task, types = c("tfd_irreg", "tfd_reg")) {
      ft <- task$feature_types
      ft[type %in% types, id]
    }

  )
)

register_pipeop("random_effect", PipeOpRandomEffect)


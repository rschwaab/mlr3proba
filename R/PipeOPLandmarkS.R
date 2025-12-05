PipeOpLandmarkLong <- R6::R6Class(
  "PipeOpLandmarkLong",
  inherit = mlr3pipelines::PipeOp,

  public = list(
    initialize = function(id = "proba.landmark_long") {
      super$initialize(
        id = id,
        param_set = paradox::ps(
          landmark_time = paradox::p_dbl(lower = 0, tags = c("train", "predict")),
          drop_empty    = paradox::p_lgl(init = TRUE, tags = c("train", "predict")),
          min_points    = paradox::p_int(lower = 1, init = 1, tags = c("train", "predict")),
          # long-table config (wide long: one row per id/time, columns = features)
          long          = paradox::p_uty(tags = c("train", "predict")),
          long_id_col   = paradox::p_uty(init = "id", tags = c("train", "predict")),
          long_time_col = paradox::p_uty(init = "fuptime", tags = c("train", "predict")),
          long_features = paradox::p_uty(tags = c("train", "predict"))
        ),
        input  = data.table::data.table(
          name   = "input",
          train  = "Task",
          predict= "Task"
        ),
        # Two outputs: filtered Task, and filtered Long table
        output = data.table::data.table(
          name   = c("task", "long"),
          train  = c("Task",  "Long"),
          predict= c("Task",  "Long")
        ),
        packages = c("mlr3", "mlr3proba", "data.table", "checkmate")
      )
    }
  ),

  private = list(

    .train = function(inputs) {
      task <- inputs[[1L]]
      pv   <- self$param_set$values
      # validate pv
      checkmate::assert_string(pv$long_id_col)
      checkmate::assert_string(pv$long_time_col)
      checkmate::assert_number(pv$landmark_time, lower = 0, finite = TRUE)

      idcol <- private$.get_id_col(task)
      long <- private$.get_long(pv$long)
      long_feats <- private$.get_long_features(pv$long_features, long, pv$long_id_col, pv$long_time_col)
      res <- private$.apply_landmark(task, idcol, long, long_feats, pv)
      list(res$task_out, res$long_out)
    },

    .predict = function(inputs) {
      task <- inputs[[1L]]
      st   <- self$state
      if (is.null(st$landmark_time)) stop("PipeOpLandmarkLong was not trained.")

      pv         <- self$param_set$values
      long       <- private$.get_long(pv$long)  # supply (possibly new) long at predict
      res <- private$.apply_landmark(task, st$id_col, long, st$long_id, st$long_time,
                                     st$long_features, st$landmark_time,
                                     st$drop_empty, st$min_points)
      list(res$task_out, res$long_out)
    },

    # ---- core ----
    .apply_landmark = function(task, idcol, long, long_feats, pv) {
      if (!is.null(pv$min_points)) checkmate::assert_int(pv$min_points, lower = 1)

      # task data & at-risk filtering
      keep_cols <- unique(c(idcol, task$feature_names, task$target_names))
      dt <- data.table::as.data.table(task$data(cols = keep_cols))
      time_name  <- task$target_names[1L]
      event_name <- task$target_names[2L]

      dt <- dt[get(time_name) > pv$landmark_time]
      if (nrow(dt) == 0L) stop("No subjects remain at risk at landmark_time = ", pv$landmark_time, ".")

      ord_ids <- as.character(dt[[idcol]])

      # prep long
      L <- data.table::as.data.table(long)
      if (!all(c(pv$long_id_col, pv$long_time_col) %in% names(L))) stop("long_id_col/long_time_col not found in 'long'.")

      miss_feats <- setdiff(long_feats, names(L))
      if (length(miss_feats)) stop("Missing long_features in 'long': ", paste(miss_feats, collapse = ", "))

      # normalize id/time, filter ids and time <= LM
      L[, (pv$long_id_col) := as.character(get(pv$long_id_col))]
      L <- L[get(pv$long_id_col) %in% ord_ids]
      L <- L[!is.na(get(pv$long_time_col)) & get(pv$long_time_col) <= pv$landmark_time]
      if (nrow(L) == 0L) stop("No rows remain in 'long' after filtering by ids and time <= landmark_time.")

      # strict policy across features: require >= min_points per feature per id
      ids_keep <- ord_ids
      for (nm in long_feats) {
        cnt <- L[!is.na(get(nm)), .N, by = c(pv$long_id_col)]
        have <- cnt[N >= pv$min_points, get(pv$long_id_col)]
        ids_keep <- intersect(ids_keep, as.character(have))
      }
      if (isTRUE(pv$drop_empty)) {
        if (!length(ids_keep))
          stop("After filtering by min_points across features, no ids remain.")
        dt <- dt[as.character(get(idcol)) %in% ids_keep]
        L  <- L[get(pv$long_id_col) %in% ids_keep]
      }

      # rebase survival time (maybe consider also rebasing L)
      dt[, (time_name) := get(time_name) - pv$landmark_time]

      # rebuild TaskSurv & restore roles
      backend <- mlr3::as_data_backend(dt)
      task_out <- mlr3proba::TaskSurv$new(
        id      = task$id,
        backend = backend,
        time    = time_name,
        event   = event_name
      )
      task_out$col_roles$group   <- idcol
      task_out$col_roles$feature <- setdiff(task_out$col_roles$feature, idcol)

      # long output (id, time, selected features), ordered
      browser()
      cols <- c(pv$long_id_col, pv$long_time_col, long_feats)
      long_out <- L[, ..cols]                     # or: L[, cols, with = FALSE]
      data.table::setorderv(long_out, c(pv$long_id_col, pv$long_time_col))
      long_out <- structure(long_out, class = c("Long", class(long_out)))

      list(task_out = task_out, long_out = long_out)
    },

    # ---- helpers ----
    .get_id_col = function(task) {
      id_col <- task$col_roles$group
      if (length(id_col) == 0L) stop("No group column set on task.")
      if (length(id_col) != 1L) stop("Need exactly one group column.")
      id_col[[1L]]
    },

    .get_long = function(long) {
      if (is.null(long)) stop("Parameter 'long' is NULL. Provide your long table via param 'long'.")
      if (!inherits(long, c("data.frame", "data.table")))
        stop("Parameter 'long' must be a data.frame/data.table.")
      data.table::as.data.table(long)
    },

    .get_long_features = function(long_features, long, long_id, long_time_col) {
      if (is.null(long_features)) {
        # auto-detect numeric feature columns (excluding common id/time names)
        num_cols <- names(long)[vapply(long, is.numeric, FUN.VALUE = logical(1))]
        num_cols <- setdiff(num_cols, c(long_id, long_time_col))
        if (!length(num_cols)) stop("Could not auto-detect numeric long features. Set 'long_features'.")
        return(num_cols)
      }
      checkmate::assert_character(long_features, min.chars = 1)
      long_features
    }
  )
)

register_pipeop("landmark_long", PipeOpLandmarkLong)

# --------------------- PipeOp: Landmark on long table ---------------------
PipeOpLandmarkLong <- R6::R6Class(
  "PipeOpLandmarkLong",
  inherit = mlr3pipelines::PipeOp,

  public = list(
    initialize = function(id = "proba.landmark_long") {
      super$initialize(
        id = id,
        param_set = paradox::ps(
          landmark_time = paradox::p_dbl(lower = 0, tags = c("train","predict")),
          drop_empty    = paradox::p_lgl(init = TRUE, tags = c("train","predict")),
          min_points    = paradox::p_int(lower = 1, init = 1, tags = c("train","predict")),
          long          = paradox::p_uty(tags = c("train","predict")),
          long_id_col   = paradox::p_uty(init = "id",      tags = c("train","predict")),
          long_time_col = paradox::p_uty(init = "fuptime", tags = c("train","predict")),
          long_features = paradox::p_uty(tags = c("train","predict"))
        ),
        input  = data.table::data.table(
          name   = "input",
          train  = "Task",
          predict= "Task"
        ),
        # Two outputs: Task and Long (custom type).
        output = data.table::data.table(
          name   = c("task","long"),
          train  = c("Task","Long"),
          predict= c("Task","Long")
        ),
        packages = c("mlr3","mlr3proba","data.table","checkmate")
      )
    }
  ),

  private = list(

    .train = function(inputs) {
      task <- inputs[[1L]]
      pv   <- self$param_set$values

      checkmate::assert_string(pv$long_id_col)
      checkmate::assert_string(pv$long_time_col)
      checkmate::assert_number(pv$landmark_time, lower = 0, finite = TRUE)
      if (!is.null(pv$min_points)) checkmate::assert_int(pv$min_points, lower = 1)

      idcol      <- private$.get_id_col(task)
      long_full  <- private$.get_long(pv$long)
      long_feats <- private$.get_long_features(pv$long_features, long_full, pv$long_id_col, pv$long_time_col)

      res <- private$.apply_landmark(task, idcol, long_full, long_feats, pv)

      # store config for predict
      self$state <- list(
        id_col        = idcol,
        long_id_col   = pv$long_id_col,
        long_time_col = pv$long_time_col,
        long_features = long_feats,
        landmark_time = pv$landmark_time,
        drop_empty    = pv$drop_empty,
        min_points    = pv$min_points
      )
      list(res$task_out, res$long_out)
    },

    .predict = function(inputs) {
      task <- inputs[[1L]]
      st   <- self$state
      if (is.null(st$landmark_time)) stop("PipeOpLandmarkLong was not trained.")

      pv <- self$param_set$values
      long_full <- private$.get_long(pv$long)  # allow new long at predict

      # reuse train-time feature set & columns
      pv2 <- list(
        landmark_time = st$landmark_time,
        drop_empty    = st$drop_empty,
        min_points    = st$min_points,
        long_id_col   = st$long_id_col,
        long_time_col = st$long_time_col
      )
      class(pv2) <- "list"

      res <- private$.apply_landmark(task, st$id_col, long_full, st$long_features, pv2)
      list(res$task_out, res$long_out)
    },

    .apply_landmark = function(task, idcol, long_full, long_feats, pv) {
      # baseline/task subset & at-risk filter
      keep_cols <- unique(c(idcol, task$feature_names, task$target_names))
      dt <- data.table::as.data.table(task$data(cols = keep_cols))
      time_name  <- task$target_names[1L]
      event_name <- task$target_names[2L]

      dt <- dt[get(time_name) > pv$landmark_time]
      if (nrow(dt) == 0L) stop("No subjects remain at risk at landmark_time = ", pv$landmark_time, ".")

      ord_ids <- as.character(dt[[idcol]])

      # long subset
      L <- data.table::as.data.table(long_full)
      if (!all(c(pv$long_id_col, pv$long_time_col) %in% names(L)))
        stop("long_id_col/long_time_col not found in 'long'.")

      miss_feats <- setdiff(long_feats, names(L))
      if (length(miss_feats)) stop("Missing long_features in 'long': ", paste(miss_feats, collapse = ", "))

      # normalize/filter long to ids & times <= landmark
      L[, (pv$long_id_col) := as.character(get(pv$long_id_col))]
      L <- L[get(pv$long_id_col) %in% ord_ids]
      L <- L[!is.na(get(pv$long_time_col)) & get(pv$long_time_col) <= pv$landmark_time]
      if (nrow(L) == 0L) stop("No rows remain in 'long' after filtering ids/time.")

      # require >= min_points per feature per id (strict across all features)
      ids_keep <- ord_ids
      for (nm in long_feats) {
        cnt  <- L[!is.na(get(nm)), .N, by = list(get(pv$long_id_col))]
        data.table::setnames(cnt, "get", pv$long_id_col) # rename the auto "get" back to id col
        have <- cnt[N >= pv$min_points, get(pv$long_id_col)]
        ids_keep <- intersect(ids_keep, as.character(have))
      }

      if (isTRUE(pv$drop_empty)) {
        if (!length(ids_keep)) stop("After filtering by min_points across features, no ids remain.")
        dt <- dt[as.character(get(idcol)) %in% ids_keep]
        L  <- L[get(pv$long_id_col) %in% ids_keep]
      }

      # rebase survival time
      dt[, (time_name) := get(time_name) - pv$landmark_time]

      # rebuild Task
      backend <- mlr3::as_data_backend(dt)
      task_out <- mlr3proba::TaskSurv$new(
        id      = task$id,
        backend = backend,
        time    = time_name,
        event   = event_name
      )
      task_out$col_roles$group   <- idcol
      task_out$col_roles$feature <- setdiff(task_out$col_roles$feature, idcol)

      # long output (typed as "Long")
      cols <- c(pv$long_id_col, pv$long_time_col, long_feats)
      long_out <- L[, ..cols]
      data.table::setorderv(long_out, c(pv$long_id_col, pv$long_time_col))
      class(long_out) <- c("Long", class(long_out))

      list(task_out = task_out, long_out = long_out)
    },

    .get_id_col = function(task) {
      id_col <- task$col_roles$group
      if (length(id_col) == 0L) stop("No group column set on task.")
      if (length(id_col) != 1L) stop("Need exactly one group column.")
      id_col[[1L]]
    },

    .get_long = function(long) {
      if (is.null(long)) stop("Parameter 'long' is NULL. Provide your long table via param 'long'.")
      if (!inherits(long, c("data.frame","data.table")))
        stop("Parameter 'long' must be a data.frame/data.table.")
      data.table::as.data.table(long)
    },

    .get_long_features = function(long_features, long, id_col, time_col) {
      if (is.null(long_features)) {
        num <- names(long)[vapply(long, is.numeric, FUN.VALUE = logical(1))]
        num <- setdiff(num, c(id_col, time_col))
        if (!length(num)) stop("Could not auto-detect numeric long features. Set 'long_features'.")
        return(num)
      }
      checkmate::assert_character(long_features, min.chars = 1)
      long_features
    }
  )
)

register_pipeop("landmark_long", PipeOpLandmarkLong)



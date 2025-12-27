#' @title Landmarking of a Survival Task with a Longitudinal Table
#'
#' @name mlr_pipeops_landmark_long
#' @aliases PipeOpLandmarkLong
#' @description
#' Creates a *landmark dataset* from a survival task (`TaskSurv`) and a subject-level longitudinal
#' table (`long`). At a given `landmark_time`, the PipeOp:
#' \itemize{
#'   \item keeps only subjects still at risk (`time > landmark_time`),
#'   \item rebases survival time by subtracting `landmark_time`,
#'   \item filters longitudinal rows to those subjects and measurements taken at or before the landmark,
#'   \item (optionally) drops subjects without enough longitudinal information (`min_points`).
#' }
#'
#' The filtered longitudinal table is returned on a second output channel and is given class `"Long"`.
#'
#' @details
#' This PipeOp expects the input task to have exactly one group column (subject identifier) set via
#' `task$col_roles$group`. The survival target is assumed to be in the order returned by
#' `task$target_names`, where the first target is the time column and the second target is the event
#' column.
#'
#' If `long_features` is `NULL`, numeric longitudinal feature columns are auto-detected from `long`
#' (excluding `long_id_col` and `long_time_col`).
#'
#' @section Parameters:
#' \describe{
#' \item{landmark_time (`numeric(1)`)}{Landmark time \eqn{t_L \ge 0}. Subjects with `time <= landmark_time`
#' are removed and remaining survival times are rebased by subtracting `landmark_time`. Must be set.}
#' \item{drop_empty (`logical(1)`)}{If `TRUE` (default), subjects that do not meet the `min_points`
#' requirement for *any* longitudinal feature are removed from both the task and the longitudinal output.}
#' \item{min_points (`integer(1)`)}{Minimum number of *non-missing* longitudinal measurements required
#' per subject and per longitudinal feature (default: `1`). The requirement is enforced feature-wise and
#' combined strictly across all `long_features` (intersection over features).}
#' \item{long (`data.frame` | `data.table`)}{Longitudinal table. Must contain `long_id_col`, `long_time_col`,
#' and all `long_features`. Must be provided at train-time; may be replaced at predict-time.}
#' \item{long_id_col (`character(1)`)}{Column name in `long` holding the subject identifier (default: `"id"`).}
#' \item{long_time_col (`character(1)`)}{Column name in `long` holding the measurement time (default: `"fuptime"`).}
#' \item{long_features (`character`)}{Names of longitudinal feature columns in `long`. If `NULL`,
#' numeric columns are auto-detected (excluding id/time columns).}
#' }
#'
#' @section Input and Output Channels:
#' \describe{
#' \item{Input}{`"input"`: a [`mlr3::Task`], typically a [`mlr3proba::TaskSurv`].}
#' \item{Output}{`"task"`: a rebased [`mlr3proba::TaskSurv`].\cr
#' `"long"`: a filtered `data.table` with class `"Long"` containing columns
#' `long_id_col`, `long_time_col`, and `long_features`, ordered by id and time.}
#' }
#'
#' @examples
#' library(mlr3)
#' library(mlr3proba)
#' library(mlr3pipelines)
#' library(data.table)
#'
#' # Example survival task
#' dt <- data.table(
#'   id = 1:5,
#'   x1 = rnorm(5),
#'   time = c(10, 8, 12, 3, 20),
#'   status = c(1, 0, 1, 1, 0)
#' )
#' task <- TaskSurv$new("toy", backend = dt, time = "time", event = "status")
#' task$col_roles$group <- "id"
#'
#' # Example long table (multiple rows per id)
#' long <- data.table(
#'   id = rep(1:5, each = 3),
#'   fuptime = rep(c(1, 4, 7), times = 5),
#'   biomarker = rnorm(15),
#'   lab = rnorm(15)
#' )
#'
#' pop <- po("landmark_long",
#'   landmark_time = 5,
#'   long = long,
#'   long_id_col = "id",
#'   long_time_col = "fuptime",
#'   long_features = c("biomarker", "lab"),
#'   min_points = 1L,
#'   drop_empty = TRUE
#' )
#'
#' out <- pop$train(list(task))
#' out$task  # rebased TaskSurv
#' out$long  # filtered longitudinal table (class "Long")


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



#' @title Survival Pipeline: Landmarking + Random Effects from Longitudinal Data
#'
#' @name mlr_graphs_landmark_randomeffects_surv
#' @aliases pipeline_landmark_randomeffects_surv
#' @description
#' Creates an \pkg{mlr3pipelines} graph / (optionally) a \pkg{mlr3} GraphLearner that:
#' \enumerate{
#'   \item applies landmarking to a survival task and a longitudinal table via `po("landmark_long")`,
#'   \item extracts subject-specific random intercept/slope features from the longitudinal table via
#'         `po("random_effect_long")`,
#'   \item trains/predicts with a provided survival learner.
#' }
#'
#' The pipeline is registered as `"landmark_randomeffects_surv"`.
#'
#' @details
#' The pipeline expects survival tasks where the subject identifier is stored as the single group column
#' (`task$col_roles$group`). The longitudinal table is provided via the `long` argument and is passed to
#' `po("landmark_long")` (and then forwarded to `po("random_effect_long")`).
#'
#' Landmarking keeps only subjects still at risk at `landmark_time` (i.e. `time > landmark_time`) and
#' rebases the survival time by subtracting `landmark_time`. The longitudinal table is filtered to
#' those subjects and rows with measurement time `<= landmark_time`. Optionally, subjects without at
#' least `min_points` non-missing measurements per longitudinal feature can be dropped (`drop_empty`).
#'
#' Random-effects extraction fits, for each selected longitudinal feature, a linear mixed model
#' \deqn{y \sim t + (1 + t \mid id)}
#' and appends the per-subject random intercept and random slope as new features to the task.
#'
#' @section Returned object:
#' If `graph_learner = FALSE` (default), returns a [`mlr3pipelines::Graph`].\cr
#' If `graph_learner = TRUE`, returns a [`mlr3pipelines::GraphLearner`] via `create_grlrn()`.
#'
#' @section Graph layout:
#' \preformatted{
#' landmark_long  --(task)-->  random_effect_long  --(output)-->  learner
#' landmark_long  --(long)-->  random_effect_long
#' }
#'
#' @param learner ([`mlr3::Learner`])\cr
#'   Survival learner to be wrapped (must support task type `"surv"`). A deep clone is made internally.
#' @param long (`data.frame` | `data.table`)\cr
#'   Longitudinal table passed to `po("landmark_long")`. Must contain `long_id_col`, `long_time_col`,
#'   and the required feature columns.
#' @param landmark_time (`numeric(1)`)\cr
#'   Landmark time \eqn{t_L \ge 0}. Default: `0`.
#' @param long_id_col (`character(1)`)\cr
#'   Subject id column name in `long`. Default: `"id"`.
#' @param long_time_col (`character(1)`)\cr
#'   Measurement time column name in `long`. Default: `"fuptime"`.
#' @param min_points (`integer(1)`)\cr
#'   Minimum number of non-missing measurements per subject and longitudinal feature required by the
#'   landmarking step. Default: `1`.
#' @param drop_empty (`logical(1)`)\cr
#'   Whether to drop subjects that do not meet `min_points` for all landmarked longitudinal features.
#'   Default: `TRUE`.
#' @param long_features (`character`)\cr
#'   Longitudinal feature columns used in the landmarking step. If `NULL`, numeric columns are
#'   auto-detected by `po("landmark_long")`. Default: `NULL`.
#' @param re_features (`character`)\cr
#'   Longitudinal feature columns modeled by the random-effects step. If `NULL`, numeric columns are
#'   auto-detected by `po("random_effect_long")`. Default: `NULL`.
#' @param graph_learner (`logical(1)`)\cr
#'   If `TRUE`, wrap the graph in a `GraphLearner` using `create_grlrn()`. Default: `FALSE`.
#'
#' @examples
#' library(mlr3)
#' library(mlr3proba)
#' library(mlr3pipelines)
#' library(data.table)
#'
#' set.seed(1)
#'
#' # toy TaskSurv (50 subjects)
#' dt <- data.table(
#'   id     = 1:50,
#'   x1     = rnorm(50),
#'   time   = rexp(50, rate = 0.1),
#'   status = rbinom(50, 1, 0.6)
#' )
#'
#' task <- TaskSurv$new("toy", backend = dt, time = "time", event = "status")
#' task$col_roles$group <- "id"
#'
#' # toy longitudinal table (5 measurements per subject)
#' long <- data.table(
#'   id        = rep(1:50, each = 5),
#'   fuptime   = rep(seq(0, 8, length.out = 5), times = 50),
#'   biomarker = rnorm(50 * 5),
#'   lab       = rnorm(50 * 5)
#' )
#'
#' # Instantiate the registered pipeline (GraphLearner)
#' pipeline <- ppl(
#'   "landmark_randomeffects_surv",
#'   learner       = lrn("surv.coxph", predict_type = "distr"),
#'   long          = long,
#'   landmark_time = 7,
#'   long_id_col   = "id",
#'   long_time_col = "fuptime",
#'   min_points    = 2,
#'   drop_empty    = TRUE,
#'   graph_learner = TRUE
#' )
#'
#' # Train the pipeline
#' pipeline$train(task)


pipeline_landmark_randomeffects_surv = function(
    learner,
    long,
    landmark_time = 0,
    long_id_col   = "id",
    long_time_col = "fuptime",
    min_points    = 1,
    drop_empty    = TRUE,
    long_features = NULL,
    re_features   = NULL,
    graph_learner = FALSE
) {
  assert_learner(learner, task_type = "surv")
  learner = learner$clone(deep = TRUE)

  po_lm <- po("landmark_long",
              landmark_time = landmark_time,
              long          = long,
              long_id_col   = long_id_col,
              long_time_col = long_time_col,
              min_points    = min_points,
              drop_empty    = drop_empty,
              long_features = long_features
  )

  po_re <- po("random_effect_long",
              long_id_col   = long_id_col,
              long_time_col = long_time_col,
              feature_cols  = re_features
  )

  po_lr <- po("learner", learner, id = learner$id)  # id == learner$id, like your example

  gr <- Graph$new()$
    add_pipeop(po_lm)$
    add_pipeop(po_re)$
    add_pipeop(po_lr)$
    add_edge(po_lm$id, po_re$id, src_channel = "task",   dst_channel = "task")$
    add_edge(po_lm$id, po_re$id, src_channel = "long",   dst_channel = "long")$
    add_edge(po_re$id, po_lr$id, src_channel = "output", dst_channel = "input")

  create_grlrn(gr, graph_learner)
}


register_graph("landmark_randomeffects_surv", pipeline_landmark_randomeffects_surv)


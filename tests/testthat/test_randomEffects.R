test_that("PipeOpRandomEffect train and predict match on same data", {
  # required packages
  library(mlr3)
  library(mlr3fda)
  library(mlr3proba)
  library(mlr3pipelines)
  library(data.table)
  library(pencal)
  library(tf)

  # create task
  data("pbc2data", package = "pencal")
  baseline <- pbc2data$baselineInfo
  long     <- pbc2data$longitudinalInfo

  # create backend data.table with tfd() longitudinal features
  pbc2_backend <- data.table(
    subject_id   = as.factor(baseline$id),
    time         = baseline$time,
    event        = baseline$event,
    baselineAge  = baseline$baselineAge,
    sex          = baseline$sex,
    treatment    = baseline$treatment,
    serBilir     = tfd(long, id = "id", arg = "fuptime", value = "serBilir"),
    albumin      = tfd(long, id = "id", arg = "fuptime", value = "albumin"),
    alkaline     = tfd(long, id = "id", arg = "fuptime", value = "alkaline"),
    SGOT         = tfd(long, id = "id", arg = "fuptime", value = "SGOT"),
    platelets    = tfd(long, id = "id", arg = "fuptime", value = "platelets"),
    prothrombin  = tfd(long, id = "id", arg = "fuptime", value = "prothrombin")
  )

  # remove any rows with missing baseline / functional entries
  pbc2_backend <- stats::na.omit(pbc2_backend)

  # to keep the test fast work on a subset
  pbc2_backend <- pbc2_backend[1:100, ]

  backend <- mlr3::as_data_backend(pbc2_backend)
  task    <- mlr3proba::TaskSurv$new(
    id      = "pbc2",
    backend = backend,
    time    = "time",
    event   = "event"
  )

  # Train and predict with PipeOpRandomEffect on the SAME task

  po_fre <- mlr3pipelines::po("random_effect")

  task_train <- po_fre$train(list(task))[[1L]]
  task_pred  <- po_fre$predict(list(task))[[1L]]

  dt_train <- task_train$data()
  dt_pred  <- task_pred$data()

  # Should match
  expect_identical(names(dt_train), names(dt_pred))
  expect_equal(dt_train, dt_pred, tolerance = 1e-6)


})

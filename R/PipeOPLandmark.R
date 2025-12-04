# Landmarking PipeOp: strict landmarking + at-risk filtering + target rebasing
PipeOpLandmark <- R6::R6Class(
  "PipeOpLandmark",
  inherit = mlr3pipelines::PipeOp,

  public = list(
    initialize = function(id = "proba.landmark") {
      super$initialize(
        id = id,
        param_set = paradox::ps(
          landmark_time = paradox::p_dbl(lower = 0, tags = c("train", "predict")),
          strict        = paradox::p_lgl(init = TRUE, tags = c("train", "predict")),  # reserved for future use
          drop_empty    = paradox::p_lgl(init = TRUE, tags = c("train", "predict"))
        ),
        input  = data.table::data.table(name = "input",  train = "Task", predict = "Task"),
        output = data.table::data.table(name = "output", train = "Task", predict = "Task"),
        packages = c("mlr3", "mlr3proba", "mlr3fda", "data.table", "tf", "checkmate")
      )
    }
  ),

  private = list(

    .train = function(inputs) {
      task <- inputs[[1L]]

      lm_t <- self$param_set$values$landmark_time
      checkmate::assert_number(lm_t, lower = 0, finite = TRUE)

      idcol <- private$.get_id_col(task)
      funs  <- private$.tfd_cols(task)

      new_task <- private$.landmark_transform(task, lm_t, idcol, funs)

      self$state <- list(
        landmark_time = lm_t,
        id_col        = idcol,
        fun_cols      = funs
      )
      list(new_task)
    },

    .predict = function(inputs) {
      task <- inputs[[1L]]
      st   <- self$state
      if (is.null(st$landmark_time)) stop("PipeOpLandmark was not trained.")
      new_task <- private$.landmark_transform(task, st$landmark_time, st$id_col, st$fun_cols)
      list(new_task)
    },

    .landmark_transform = function(task, lm_t, idcol, fun_cols) {
      if (length(fun_cols) == 0L) stop("No functional (tfd_*) columns found.")

      drop_empty <- isTRUE(self$param_set$values$drop_empty)

      # pull required columns (id + features + target)
      keep_cols <- unique(c(idcol, task$feature_names, task$target_names))
      dt <- data.table::as.data.table(task$data(cols = keep_cols))

      time_name  <- task$target_names[1L]
      event_name <- task$target_names[2L]

      # 1) keep only subjects still at risk at landmark time
      dt <- dt[get(time_name) > lm_t]
      if (nrow(dt) == 0L) stop("No subjects remain at risk at landmark_time = ", lm_t, ".")

      # 2) truncate each tfd feature at arg <= lm_t and gather IDs with any pre-LM data
      ids_keep_fun <- as.character(dt[[idcol]])
      tabs_trunc <- setNames(vector("list", length(fun_cols)), fun_cols)

      for (nm in fun_cols) {
        x   <- dt[[nm]]
        tab <- as.data.frame(x, unnest = TRUE)
        tab <- tab[!is.na(tab$arg) & !is.na(tab$value) & tab$arg <= lm_t, , drop = FALSE]
        stopifnot(all(tab$arg <= lm_t + 1e-12, na.rm = TRUE))

        ids_with_data <- unique(as.character(tab$id))
        ids_keep_fun  <- intersect(ids_keep_fun, ids_with_data)   # strict policy

        # store truncated long table; assignment happens after filtering/reordering
        tabs_trunc[[nm]] <- tab
      }

      # 3) drop subjects without pre-LM data for ALL functional features (strict)
      if (drop_empty) {
        dt <- dt[as.character(get(idcol)) %in% ids_keep_fun]
        if (nrow(dt) == 0L) {
          stop("After strict landmarking, no subject has pre-landmark data for all functional features.")
        }
      }

      # 4) rebuild each tfd column, aligned & named to match dt row order
      ord_ids <- as.character(dt[[idcol]])

      for (nm in fun_cols) {
        tab <- tabs_trunc[[nm]]
        tab$id <- as.character(tab$id)
        tab    <- tab[tab$id %in% ord_ids, , drop = FALSE]
        tab    <- tab[order(match(tab$id, ord_ids), tab$arg), , drop = FALSE]

        col <- tf::tfd(tab, id = "id", arg = "arg", value = "value")

        # ensure names exist; then reorder and force final names to ord_ids
        nms <- names(col)
        if (is.null(nms)) nms <- unique(tab$id)
        idx <- match(ord_ids, nms)
        if (anyNA(idx)) {
          stop("Missing/unnamed tfd elements for ids: ",
               paste(ord_ids[is.na(idx)], collapse = ", "))
        }
        col <- col[idx]
        names(col) <- ord_ids

        if (length(col) != nrow(dt)) {
          stop(sprintf("Internal error: length mismatch for %s: %d vs %d",
                       nm, length(col), nrow(dt)))
        }

        # sanity check mirroring RE's na.omit view
        un <- stats::na.omit(as.data.frame(col, unnest = TRUE))
        miss <- setdiff(ord_ids, unique(as.character(un$id)))
        if (length(miss)) {
          stop(sprintf("Strict LM check: '%s' has no pre-LM data for ids: %s",
                       nm, paste(head(miss, 5), collapse = ", ")))
        }

        dt[[nm]] <- col
      }

      # 5) rebase survival time to time since landmark
      dt[, (time_name) := get(time_name) - lm_t]

      # 6) rebuild TaskSurv and restore roles
      backend <- mlr3::as_data_backend(dt)
      new_task <- mlr3proba::TaskSurv$new(
        id      = task$id,
        backend = backend,
        time    = time_name,
        event   = event_name
      )
      new_task$col_roles$group   <- idcol
      new_task$col_roles$feature <- setdiff(new_task$col_roles$feature, idcol)

      new_task
    },

    .get_id_col = function(task) {
      id_col <- task$col_roles$group
      if (length(id_col) == 0L) stop("No group column set on task.")
      if (length(id_col) != 1L) stop("Need exactly one group column.")
      id_col[[1L]]
    },

    .tfd_cols = function(task, types = c("tfd_irreg", "tfd_reg")) {
      ft <- task$feature_types
      ft[type %in% types, id]
    }
  )
)

register_pipeop("landmark", PipeOpLandmark)

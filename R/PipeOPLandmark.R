PipeOpLandmark <- R6::R6Class(
  "PipeOpLandmark",
  inherit = mlr3pipelines::PipeOp,

  public = list(
    initialize = function(id = "proba.landmark") {
      super$initialize(
        id = id,
        param_set = paradox::ps(
          landmark_time = paradox::p_dbl(lower = 0, tags = c("train", "predict")),
          strict        = paradox::p_lgl(init = TRUE, tags = c("train", "predict")),
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

      drop_empty <- self$param_set$values$drop_empty

      # pull required columns
      keep_cols <- unique(c(idcol, task$feature_names, task$target_names))
      dt <- data.table::as.data.table(task$data(cols = keep_cols))

      time_name  <- task$target_names[1L]
      event_name <- task$target_names[2L]

      # 1) Filter at-risk
      dt <- dt[get(time_name) > lm_t]
      if (nrow(dt) == 0L) stop("No subjects remain at risk at landmark_time = ", lm_t, ".")

      ids_keep_fun <- as.character(dt[[idcol]])
      tabs_trunc <- setNames(vector("list", length(fun_cols)), fun_cols)

      for (nm in fun_cols) {
        x   <- dt[[nm]]

        # Ensure x names match the ID column to prevent ID misalignment
        # This is a crucial safety step if row order changed
        if (is.null(names(x))) names(x) <- as.character(dt[[idcol]])

        tab <- as.data.frame(x, unnest = TRUE)

        # 2) Strict Filter: Remove NAs and truncate time
        tab <- tab[!is.na(tab$arg) & !is.na(tab$value) & tab$arg <= lm_t, , drop = FALSE]

        # --- NEW STEP 2.5: SANITIZE ARGUMENTS ---
        # 1. Round to 6 decimal places to fix floating point jitter
        tab$arg <- round(tab$arg, 6)

        # 2. Check for duplicates. If a subject has two values at t=0.5, keep the first one.
        # This prevents the "(Almost) non-unique" error.
        is_dup <- duplicated(tab[, c("id", "arg")])
        if (any(is_dup)) {
          warning(sprintf("Removed %d duplicate time points for feature %s (e.g. ID %s at t=%s)",
                          sum(is_dup), nm, tab$id[which(is_dup)[1]], tab$arg[which(is_dup)[1]]))
          tab <- tab[!is_dup, , drop = FALSE]
        }
        # ----------------------------------------

        ids_with_data <- unique(as.character(tab$id))
        ids_keep_fun  <- intersect(ids_keep_fun, ids_with_data)
        tabs_trunc[[nm]] <- tab
      }

      # 3) Drop empty subjects
      if (drop_empty) {
        dt <- dt[as.character(get(idcol)) %in% ids_keep_fun]
        if (nrow(dt) == 0L) stop("No subjects have pre-landmark data.")
      }

      # 4) Rebuild tfd columns
      ord_ids <- as.character(dt[[idcol]])

      for (nm in fun_cols) {
        tab <- tabs_trunc[[nm]]
        tab$id <- as.character(tab$id)

        # Filter to survivors
        tab <- tab[tab$id %in% ord_ids, , drop = FALSE]

        # Order by ID then Arg to ensure clean reconstruction
        tab <- tab[order(match(tab$id, ord_ids), tab$arg), , drop = FALSE]

        # Rebuild
        col <- tf::tfd(tab, id = "id", arg = "arg", value = "value")

        # Align names strictly
        # tf::tfd might return names in the order of 'tab', which we just sorted to match 'ord_ids'
        # But we double check with match()
        nms <- names(col)
        if (is.null(nms)) nms <- unique(tab$id)

        idx <- match(ord_ids, nms)
        if (anyNA(idx)) stop("Internal Error: Mismatch between Data ID and TFD ID during reconstruction")

        col <- col[idx]
        names(col) <- ord_ids # Force names

        dt[[nm]] <- col
      }

      # 5) Rebase Time
      dt[, (time_name) := get(time_name) - lm_t]

      # 6) Finalize
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

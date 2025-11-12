PipeOpRandomEffect <- R6::R6Class(
  "PipeOpRandomEffect",
  inherit = mlr3pipelines::PipeOp,
  public = list(
    initialize = function(id = "fda.random_effect") {
      super$initialize(
        id = id,
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
      st <- self$state
      fun_cols <- st$fun_cols
      models <- st$models
      # check if there are functional columns
      if (length(fun_cols) == 0L) {
        return(list(task))
      }
      dt_fun <- task$data(cols = fun_cols)

      feat_pieces <- vector(mode = "list" ,length = length(fun_cols))
      names(feat_pieces) <- fun_cols

      # PRC formula: û_i = D Z_i^T V_i^{-1}(y_i − X_i β)
      for (nm in fun_cols){
        x <- dt_fun[[nm]]
        tab <- as.data.frame(x, unnest = TRUE)
        tab <- stats::na.omit(tab)

        # extract LMM parameters from training fit
        mod <- models[[nm]]
        beta <- lme4::fixef(mod)
        # double check this
        D <- as.matrix(lme4::VarCorr(mod)$id)
        sigma2 <- lme4::getME(mod, "sigma")^2

        ids <- unique(tab$id)
        n_id <- length(ids)

        u_hat <- matrix(NA_real_, nrow = n_id, ncol = 2L,
                        dimnames = list(as.character(ids),
                        c("random_intercept", "random_slope")))

        for (j in seq_along(ids)){
          id_j <- ids[j]
          dat_j <- tab[tab$id == id_j, ]

          y <- dat_j$value
          arg <- dat_j$arg

          X <- cbind(1, arg)
          Z <- X
          residual <- y - X %*% beta
          V <- Z %*% D %*% t(Z) + sigma2 * diag(nrow(Z))
          # solve(V, residual) = V^{-1} %*% residual
          u_hat[j, ] <- as.numeric(D %*% t(Z) %*% solve(V, residual))
        }
        feats <- data.table::as.data.table(u_hat)
        feats[, id := as.integer(rownames(u_hat))]
        data.table::setorder(feats, id)
        feats[, id := NULL]

        data.table::setnames(
          feats,
          sprintf("%s_%s", nm, c("random_intercept", "random_slope"))
        )

        feat_pieces[[nm]] <- feats
      }
      feat_dt <- do.call(cbind, unname(feat_pieces))
      dt_keep <- task$data(cols = c(setdiff(task$feature_names, fun_cols), task$target_names))

      dt_new <- data.table::as.data.table(cbind(dt_keep, feat_dt))
      stopifnot(nrow(dt_new) == task$nrow)

      backend <- mlr3::as_data_backend(dt_new)
      new_task <- mlr3proba::TaskSurv$new(
        id      = task$id,
        backend = backend,
        time    = task$target_names[1L],
        event   = task$target_names[2L]
      )

      list(new_task)
    },
    .tfd_cols = function(task, types = c("tfd_irreg", "tfd_reg")) {
      ft <- task$feature_types
      ft[type %in% types, id]
    }

  )
)

register_pipeop("random_effect", PipeOpRandomEffect)

# https://chatgpt.com/c/6909faa8-6ad8-8328-b931-673bf45e6766

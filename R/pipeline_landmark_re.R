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


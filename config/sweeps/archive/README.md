# 历史 Sweep 配置

这里保存已经完成的批量实验定义。展开后的独立配置位于 `generated/`，历史结果仍位于 `results/<sweep_name>/`。

如需复现实验，直接把本目录中的 Sweep YAML 传给 `scripts/run_config_sweep.py`。配置内的 `output_dir` 已指向归档后的 `generated/` 路径。

当前 TLS/Huber 统一窗口的24个单次回测配置不在这里，仍位于 `config/tls_huber_selected_24/`。

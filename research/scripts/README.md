# Research Scripts

这里保存一次性或阶段性的模型评估、图表生成和配置生成脚本。它们不属于生产回测入口。

所有脚本均以项目根目录为工作目录运行，例如：

```powershell
python research/scripts/score_beta_models.py
```

当前24个TLS/Huber统一窗口配置可通过以下脚本重新生成：

```powershell
python research/scripts/generate_tls_huber_selected_configs.py
```

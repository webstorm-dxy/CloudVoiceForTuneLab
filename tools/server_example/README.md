# FastAPI 参考服务端（tunelab-cloudvoice/1）

这是 [docs/server-guide.md](../../docs/server-guide.md) 配套的**生产骨架参考实现**：实现了协议校验、
鉴权、请求体限制、并发信号量、错误语义、WAV 响应与音素/音高/energy 回填，并用加法合成做占位后端。

**它不是一个生产级歌声合成服务**：后端只是正弦波演示。接真实模型时只改
[app.py](app.py) 里标注 `# PLUG IN YOUR MODEL HERE` 的 synthesize_block() 内部实现即可。

## 运行

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python app.py --host 0.0.0.0 --port 8080
# 需要鉴权时：
.venv/bin/python app.py --host 0.0.0.0 --port 8080 --api-key secret-token
```

可选参数：`--max-concurrent N`（默认 2）。

## 在 TuneLab 里接上

**设置 → 扩展 → Cloud Voice**：

| 字段 | 值 |
| --- | --- |
| Service URL | http://127.0.0.1:8080/synthesize |
| API key | 若启动时传了 `--api-key`，填同一个值；否则留空 |

示例服务端还提供 `GET /schema`（属性面板参数 / 可画线轨 / 回显轨 / 自动音高声明）；
换成自己引擎的参数后，TuneLab 属性面板与画线轨会自动跟着变。

插件加载时会自动 `GET /voices`（本示例返回 "demo" 声库），不需要手工填声库目录。
然后新建声源片段，选 "Demo" 声库，输入歌词播放即可。

## 已验证

在 macOS 上对接 TuneLab 2.1.1 无头宿主验证过：

- 单音合成 200（约 150ms），编辑器读回音素 `l` + `a`；
- 两音合成 + `project export-audio` 导出 3 秒 529KB 非空 WAV；
- 无 token → 401，带正确 token → 200；
- 错误 protocol → 400，错误文案可读。

## 生产前还需要补的

- 换成真实模型/第三方 API，并处理长块、GPU 并发与队列（见指南 §6、§9）；
- 补客户端断连的协作式取消（当前示例用 to_thread，不能中途打断推理）；
- 加真实指标/日志脱敏/限流与密钥管理（见指南 §8、§10）；
- 按你的计费与合规要求加缓存/配额/审计。
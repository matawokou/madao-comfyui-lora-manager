# MADAO ComfyUI LoRA Manager（本地增强版）

> 📖 **本 fork 7 项定制的使用教程见 [USAGE.md](USAGE.md)**。基础用法与上游一致。

> 基于 [willmiao/ComfyUI-Lora-Manager](https://github.com/willmiao/ComfyUI-Lora-Manager) 的本地定制 fork，遵循 **GPL v3** 协议开源。
> 在上游基础上做了若干实用增强，全部改动见下方清单。感谢 willmiao 的出色工作。

## 这个 fork 改了什么

| # | 改动 | 说明 |
|---|---|---|
| 1 | **LoRA 语法默认保留文件夹前缀** | `lora_syntax_format` 默认值 `legacy` → `full`。工作流加载器可按分类分组 LoRA，不再全部落进「未分类」 |
| 2 | **视频预览修复** | 预览 URL 的缓存破坏参数（`?v=…`）移到 path 参数之前。修复 mp4/webm 预览因 `endsWith('.mp4')` 检测失效而渲染成破图的问题 |
| 3 | **扫描后自动关联 CivitAI** | 每次扫描成功后自动按文件 hash 为新模型拉取 CivitAI 元数据（已关联的库零开销）。设环境变量 `LORA_MANAGER_DISABLE_AUTO_LINK=1` 可关闭 |
| 4 | **CivitAI 下载报错友好化** | 解析 CivitAI 错误响应体，「作者已禁用该文件下载」不再误报为「API Key 无效」 |
| 5 | **NSFW 模糊一键开关** | 工具栏新增 NSFW 按钮（眼睛图标），一键切换成熟内容模糊；另增按作者排序 |
| 6 | **触发词脚本增强** | `auto_trigger_words.js`：支持任意标记为「发送提示词目标」的节点（不限于已知类名）、支持触发词前置模式、识别 Text Multiline 节点 |
| 7 | **防注册表误更新** | 包名与上游刻意区分，插件内自更新端点停用——防止 ComfyUI-Manager / 绘世启动器把定制版「更新」回上游抹掉改动 |

涉及文件均以 `LOCAL FORK` 注释标记，便于与上游 diff。

## 🔗 配套项目

- **[MADAO Prompt Manager](https://github.com/matawokou/madao-prompt-manager)** —— 浏览器提示词管理扩展，与本管理器的 LoRA 节点联动：一键把提示词发送到工作流的 LoRA/正向提示词节点，右键保存提示词时自动抓取最近生成图当封面。

## 安装

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/matawokou/madao-comfyui-lora-manager.git
```

或下载 zip 解压到 `custom_nodes`。其余配置、依赖、使用方式与上游完全一致，请参考[上游文档](https://github.com/willmiao/ComfyUI-Lora-Manager#readme)。

> ⚠️ 已使用本 fork 的用户：请勿通过 ComfyUI-Manager 对它执行「更新/修复」——它的包名已与上游区分，正常情况下不会被匹配到。

## 与上游同步

本 fork 落后于上游时，可以手动合并上游改动后再覆盖本 fork 的定制点（都有 `LOCAL FORK` 标记）。欢迎提 Issue/PR。

## License

**GNU GPL v3**（继承自上游）。详见 [LICENSE](LICENSE)。

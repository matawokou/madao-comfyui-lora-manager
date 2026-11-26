# MADAO ComfyUI LoRA Manager 使用指南

> 本文档只讲**本 fork 的 7 项定制怎么用**。基础用法（界面、扫描、CivitAI 元数据、配方、下载等）与上游完全一致，请看 [willmiao/ComfyUI-Lora-Manager 官方文档](https://github.com/willmiao/ComfyUI-Lora-Manager#readme)。

---

## 1. LoRA 语法保留文件夹前缀

**作用**：往工作流 LoRA 节点写 `<lora:...>` 时保留分类文件夹前缀，工作流加载回来后 LoRA Manager 能按分类分组，不再全部堆进「未分类」。

**设置**：插件设置页 → LoRA Syntax Format → 选 **Full**（本 fork 默认已是 Full，无需动手）。

想改回上游默认行为：设置里切回 Legacy 即可。

## 2. 扫描后自动关联 CivitAI

**作用**：每次点「扫描模型」，扫出来的新模型会**自动按文件 hash 关联 CivitAI 元数据**（封面、作者、触发词等），不用再手动逐个刷新。已关联的模型零开销。

**关闭方法**（比如想省流量）：启动 ComfyUI 前设环境变量
```
LORA_MANAGER_DISABLE_AUTO_LINK=1
```

## 3. 视频预览修复

**无需操作**——mp4/webm 格式模型的预览视频不再显示为破图（上游的缓存参数位置导致前端误判，本 fork 已修）。

## 4. CivitAI 下载报错友好化

**无需操作**——下载失败时若真实原因是「作者禁用了该文件下载」，会直接告诉你，不再误导你去检查 API Key。

## 5. NSFW 模糊一键开关

**用法**：模型库页面顶部工具栏多了 **NSFW 按钮**（眼睛图标）：
- 高亮状态 = 成熟内容已模糊（默认）
- 点一下 = 关闭模糊直接显示；再点恢复
- 另外排序下拉里新增 **按作者排序（A-Z / Z-A）**

## 6. 触发词脚本增强（auto_trigger_words）

这是 ComfyUI 画布内的辅助脚本，配合 LoRA 节点自动插入触发词。本 fork 增强：

- **支持标记节点**：任何在节点属性里标记为「发送提示词目标」（`lm_marker_role = send_prompt_target`）的节点都能接收触发词，不再限于固定几个类名
- **识别 Text Multiline** 节点（WAS Node Suite 等）
- **前置模式**：触发词插入到提示词**开头**（自动补英文逗号分隔），移除时也会连同分隔符一起删干净

## 7. 防注册表误更新（重要！）

本 fork 的包名与上游刻意不同，正常情况下 ComfyUI-Manager / 绘世启动器**不会**把它识别成上游去「更新」。

⚠️ 但如果你在 ComfyUI-Manager 里看到它并手动点了「尝试更新 / 修复」，仍可能被替换成上游版本、丢失全部定制。**请勿对它执行更新操作**；想升级就回本仓库拉最新代码。

---

## 安装 / 更新本 fork

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/matawokou/madao-comfyui-lora-manager.git
```

更新：
```bash
cd ComfyUI/custom_nodes/madao-comfyui-lora-manager
git pull
```
重启 ComfyUI 生效。

## 配套扩展

配合浏览器端的 **[MADAO Prompt Manager](https://github.com/matawokou/madao-prompt-manager)** 使用体验最佳：提示词一键发送到本管理器的 LoRA / 正向提示词节点，右键保存提示词自动抓生成图当封面。

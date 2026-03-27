# GitHub Pages 发布说明

本文档说明如何把当前仓库中的 `blog/` 静态博客发布到你的 GitHub Pages 仓库：`Yaoyuanzhou/yaoyuanzhou.github.io`。

## 当前发布策略

你已经确认采用：

- **直接替换首页**
- 站点地址：`https://yaoyuanzhou.github.io/`

也就是说，GitHub Pages 仓库根目录将直接放博客首页，而不是把博客挂到 `/blog/` 子路径。

## 当前源码与发布目录的关系

### 当前仓库中的源码目录

```text
blog/
├── index.html
├── README.md
├── data/posts.js
├── posts/
└── templates/
```

### GitHub Pages 仓库中的发布目录

```text
yaoyuanzhou.github.io/
├── index.html
├── README.md
├── .nojekyll
├── data/posts.js
├── posts/
└── templates/
```

发布时，本质上就是把 `blog/` 目录内容同步到 `yaoyuanzhou.github.io` 仓库根目录。

## 为什么要加 `.nojekyll`

因为当前博客是纯静态 HTML + JS 文件，不依赖 Jekyll。

添加 `.nojekyll` 的目的：
- 禁用 GitHub Pages 默认 Jekyll 处理
- 避免静态文件被 Jekyll 特殊规则影响
- 保持目录结构简单直接

## 一次发布的标准步骤

### 1. 在当前仓库维护博客源码

平时你继续在这里维护：
- 新文章：`blog/posts/*.html`
- 索引：`blog/data/posts.js`
- 首页：`blog/index.html`

### 2. 同步到 GitHub Pages 仓库

把 `blog/` 下这些内容同步到：
- `index.html`
- `README.md`
- `data/`
- `posts/`
- `templates/`
- `.nojekyll`

### 3. 提交并推送

在 `yaoyuanzhou.github.io` 仓库执行：

```bash
git add .
git commit -m "feat: publish paper blog"
git push origin main
```

## 以后新增论文后的发布方式

当你以后说：
- `总结并上传这篇论文`
- `把这个 paper 解读后发到博客上`

我会先把文章收录到当前仓库的 `blog/` 目录。

如果你希望**同时发布到 GitHub Pages**，你可以继续补一句：

- `顺便发布到 GitHub`
- `同步到 GitHub Pages`
- `把博客站点一起更新到首页`

这样我就会继续帮你把最新博客同步到 `yaoyuanzhou.github.io`。

## 注意事项

1. 你当前的老首页内容已经被新博客替换。
2. GitHub Pages 更新通常有几十秒到几分钟延迟。
3. 如果浏览器还显示旧页面，优先尝试强制刷新。
4. 文章路径全部使用相对路径，适合根目录部署。

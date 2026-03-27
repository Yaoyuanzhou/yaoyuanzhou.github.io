# 论文博客专区

这是一个零依赖的静态博客专区，主打 **HTML 网页上传**。

## 源码目录结构

```text
blog/
├── index.html                  # 博客首页
├── README.md                   # 使用说明
├── data/
│   └── posts.js                # 文章索引
├── posts/
│   └── trm-paper-notes.html    # 示例论文页面
└── templates/
    └── paper-template.html     # 新论文模板
```

## 你以后怎么上传论文

### 方式一：直接放 HTML（推荐）

1. 把你的论文解读整理成一个独立 HTML 文件。
2. 放到 `blog/posts/`，例如：
   - `blog/posts/2026-trm.html`
   - `blog/posts/attention-is-all-you-need.html`
3. 打开 `blog/data/posts.js`，追加一条文章信息。
4. 用浏览器打开 `blog/index.html`，就能在首页看到新文章。

### 方式二：先写 Markdown，再转 HTML

如果你先产出 Markdown，可以先转成 HTML 再放进 `blog/posts/`。

## 新增文章时要改哪里

只需要改 2 处：

1. 新增文章文件：`blog/posts/你的文章.html`
2. 更新索引：`blog/data/posts.js`

示例：

```js
{
  id: 'attention-is-all-you-need',
  title: 'Attention Is All You Need',
  date: '2026-03-27',
  year: '2026',
  tags: ['Transformer', 'NLP', '经典论文'],
  summary: 'Transformer 开山之作，核心是自注意力机制替代 RNN/CNN。',
  source: 'https://arxiv.org/abs/1706.03762',
  file: './posts/attention-is-all-you-need.html',
  readingTime: '12 分钟',
  author: 'Vaswani et al.'
}
```

## 自动化工作流

现在你还可以直接使用个人级 skill：`paper-blog-publisher`

它会把以下流程串起来：

1. 识别输入来源（HTML / Markdown / PDF / arXiv / 论文网页）
2. 需要时联动 `paper-reader` 自动解读论文
3. 生成或复用 HTML 文章页
4. 自动收录到 `blog/posts/`
5. 自动更新 `blog/data/posts.js`

以后你可以直接对我说：

- `总结并上传这篇论文`
- `把这个 paper 解读后发到博客上`
- `把这个 html 论文页收录进博客`

个人级 skill 目录：

- `/Users/yaoyuanzhou/.codeflicker/skills/paper-blog-publisher`

注意：skill 创建/更新后通常需要等待约 30 秒生效，或重启 VS Code 立即生效。

## GitHub Pages 挂载

你现在已经有个人仓库：`Yaoyuanzhou/yaoyuanzhou.github.io`。

当前采用的是 **替换首页** 的方式：把博客静态文件发布到该仓库根目录，站点地址就是：

- `https://yaoyuanzhou.github.io/`

发布后的 GitHub Pages 仓库目录结构会是：

```text
yaoyuanzhou.github.io/
├── index.html
├── README.md
├── .nojekyll
├── data/
├── posts/
└── templates/
```

如果你后续继续在当前仓库维护博客源码，那么发布时只需要把 `blog/` 目录内容同步到 GitHub Pages 仓库根目录即可。

详细步骤见：`blog/GITHUB_PAGES.md`

## 发布方式

### 本地直接看

直接双击打开 `blog/index.html` 即可。

### 作为网页发布

后续如果你想把它发布成网页，可以放到任一静态托管：
- GitHub Pages
- 公司内部静态资源服务
- Nginx 静态目录

## 当前已收录

- `TRM：用语义 Token 替代 Item ID`

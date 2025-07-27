# 📦 跨境合规数据智能体（Compliance Agent for Health）

> 一个适用于医疗场景下判断数据是否传输合规的智能体


---

## 📖 项目简介（Description）

本项目旨在解决以下问题：
- 本项目由澳门科技大学-电子科技大学跨境可信数据空间工作组提出，旨在解决数据在出入境时面临的数据风险问题。
包括以下功能：
- RoPA 评估
- 个人信息检测
- 数据去标识化
- 数据风险评估
- 数据库知识问答
- 数据风险报告生成
目标用户或适用场景
- 适用场景为律所、医院、银行等涉及数据出入境的行业，目标用户包括律师、医生、金融从业人员等行业内从业人员、以及对AI领域较为熟悉的技术开发人员。
---

## 🚀 快速开始（Getting Started）

### 🧰 环境要求

- Ubuntu >= 22.04
- Node.js >= 18
- Python >= 3.12
- docker >= 27.5.1
  


### 📦 安装步骤

```bash
# 克隆仓库
git clone https://github.com/itachiliu/ComplianceAgent.git

# 进入项目目录
cd ComplianceAgent
```

配置大模型 API Key：  
创建 `.env` 文件，并补充自己的 DeepSeek API Key，内容如下：

```env
DEEPSEEK_API_KEY=your_api_key_here
```

执行构建并启动服务：

```bash
docker-compose up --build
```

编译成功后，在浏览器访问：

```
http://localhost:8080
```


## 🗂 项目结构（Project Structure）


```text
.
├── .venv/                     # 虚拟环境目录
├── backend/                  # 后端代码
├── doc/                      # 文档目录
├── docker/                   # Docker 相关配置
├── frontend/                 # 前端代码
├── Dockerfile.backend        # 后端镜像构建文件
├── Dockerfile.frontend       # 前端镜像构建文件
├── README.md                 # 项目说明文档
├── docker-compose.yml        # Docker 编排配置
└── entrypoint.sh             # 容器启动脚本
```

## 📘 文档（Documentation）

详细文档请参阅：[docs/](./docs)

## ❓ 常见问题（FAQ）

**Q:** 如何配置大模型API Key？  
**A:** 编辑.env文件（如果没有，需要自己创建），增加一行DEEPSEEK_API_KEY="(your API Key)"。

**Q:** 是否支持 Docker 部署？  
**A:** 支持，运行 `docker-compose up` 即可，如果是首次运行，则需要编译，运行`docker-compse up --build`。

---

## 🤝 贡献指南（Contributing）

欢迎参与贡献！

1. Fork 本项目
2. 创建新分支：`git checkout -b feature/xxx`
3. 提交更改：`git commit -am '添加 xxx 功能'`
4. 推送分支：`git push origin feature/xxx`
5. 提交 Pull Request

请遵循我们的 [贡献指南](./CONTRIBUTING.md)。

## 📬 联系方式（Contact）

- 作者：[@your-github-id](https://github.com/itachiliu)
- 邮箱：itachiliuy@gmail.com
- 讨论区：待创建

---



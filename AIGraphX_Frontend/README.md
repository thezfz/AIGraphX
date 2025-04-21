**AIGraphX Frontend - 用户界面设计文档 (v1.1 - Expanded Scope - 含容器化开发环境)**

**1. 引言与指导原则**

本文档阐述了 AIGraphX 前端用户界面 (UI) 的设计与实现规划，旨在为后端 API 提供一个功能丰富、交互直观、能充分发挥后端能力的界面。我们将严格遵循与后端一致的核心原则，确保前端开发高效、健壮和可维护：

-   **简洁至上 ("大道至简"):** 界面设计应保持清晰直观，避免不必要的复杂性，但要确保核心功能（搜索、过滤、排序、详情查看、关联探索）易于访问和使用。
-   **聚焦核心价值:** 优先实现让用户能够有效利用后端提供的 **多模式搜索 (关键词、语义、混合)**、**参数化查询 (过滤、排序、分页)** 以及 **图谱关联信息**。
-   **用户体验优先:** 提供清晰、一致、响应迅速且信息丰富的交互体验，包括明确的状态反馈（加载、空结果、错误、搜索模式）。
-   **组件化开发:** 构建可复用、可组合的 UI 组件，提高开发效率和可维护性。
-   **迭代开发:** 构建坚实的基础，优先实现核心功能，逐步按需增加更高级的特性（如复杂的图可视化）。
-   **可测试性:** 设计易于测试的组件和逻辑，覆盖不同的搜索模式和参数组合。
-   **显式接口与一致性:** 与后端 API 交互遵循严格规范（**以后端 `/docs` 为准**）；组件 Props 清晰明确；代码风格统一。
-   **务实的技术选型:** 使用成熟、社区支持良好、与项目需求匹配的技术。
-   **环境一致性 (!!!):** **强烈推荐** 使用 **Podman Compose** (通过 `compose.yml` 文件) 统一管理开发环境，确保 Node.js 版本、包管理器、网络连接一致，简化启动。

**2. 核心需求 (Frontend - Expanded Scope)**

前端将提供以下核心功能，充分利用后端 v1.2+ API (**请以后端运行实例的 `/docs` 页面为 API 参数、可用过滤器和排序字段的权威来源**)：

-   **多模式搜索界面:**
    -   提供文本输入框用于输入搜索查询 (`q` 参数)。
    -   **明确的搜索模式选择:** 提供 UI 控件（如 Tabs、Radio Buttons 或 Dropdown）允许用户选择搜索模式 (`search_type` 参数)：
        -   **关键词搜索 (`search_type='keyword'`)** (通过推断的 `GET /api/v1/search/keyword/{target}` 端点)
        -   **语义搜索 (`search_type='semantic'`)** (通过推断的 `GET /api/v1/search/semantic/{target}` 端点)
        -   **混合搜索 (`search_type='hybrid'`)** (当前仅支持**论文搜索**，通过推断的 `POST /api/v1/search/hybrid/papers` 端点，利用 RRF 优化排序)
    -   包含触发搜索操作的按钮。
    * **目标实体选择:** 提供方式让用户选择是搜索**论文 (`target='papers'`)** 还是**模型 (`target='models'`)**。注意混合搜索当前仅支持 `papers`。
    -   调用后端相应的搜索端点，并传递必要的参数 (`q`, `search_type`, `target`, 分页、过滤、排序参数)。
-   **高级搜索参数化:**
    -   **过滤 (根据 `/docs` 和 `SearchFilterModel`):** 提供专门的过滤区域/面板（可折叠或侧边栏），包含以下控件：
        -   **论文过滤:**
            -   日期范围选择器: `published_after` (>=), `published_before` (<=) (ISO 8601 日期字符串 'YYYY-MM-DD')。
            -   领域过滤: `filter_area` (字符串, e.g., 'CV', 'NLP')。
        -   **模型过滤:** (注意: 当前代码分析显示，模型搜索的过滤参数似乎未在服务层显式暴露，主要在 PG Repo 处理，需进一步确认 `/docs`)
    -   **排序 (根据 `/docs` 和服务层逻辑):** 提供下拉菜单，允许用户选择排序字段 (`sort_by`) 和排序方向 (`sort_order: 'asc' | 'desc'`):
        -   **论文 (语义/混合):** `score` (默认 desc), `published_date`, `title`。
        -   **论文 (关键词):** `published_date` (默认 desc), `title` (**不支持 `score`**)。
        -   **模型 (语义):** `score` (默认 desc), `likes`, `downloads`, `last_modified`。
        -   **模型 (关键词):** `last_modified` (默认 desc), `likes`, `downloads` (**不支持 `score`**)。
    -   **分页:** 提供清晰的分页控件（如页码、上一页/下一页按钮），对应 API 参数 `page` (或 `skip`) 和 `page_size` (或 `limit`)。
    -   所有过滤、排序、分页状态需要反映在 API 请求的参数中。
-   **结果展示:**
    -   以列表形式清晰展示搜索结果。
    -   **论文列表项 (`SearchResultItem`):** 应包含 `pwc_id` (可用于链接详情), `title`, `summary`, `area`, `published_date`, `authors`, `pdf_url`，以及可选的 `score` (混合/语义搜索有值，关键词可能为 `null`)。
    -   **模型列表项 (`HFSearchResultItem`):** 应包含 `model_id` (可用于链接详情), `author`, `pipeline_tag`, `library_name`, `tags`, `likes`, `downloads`, `last_modified`，以及 `score` (仅语义搜索有值)。
    -   提供加载中 (Loading)、无结果 (No Results)、错误 (Error) 以及当前生效的搜索模式/过滤/排序条件的清晰反馈。
-   **详情查看与关联探索:**
    -   点击搜索结果项能导航到对应的详情页面 (e.g., `/papers/{pwc_id}`, `/models/{model_id}`)。
    -   **论文详情页面 (`PaperDetailResponse`):**
        -   展示从 `GET /api/v1/graph/papers/{pwc_id}` 获取的结构化元数据: `pwc_id`, `title`, `abstract`, `arxiv_id`, `url_abs` (PWC 页面), `url_pdf`, `published_date`, `authors`, `frameworks`, `number_of_stars`, `area`。
        -   **展示关联实体:** 清晰地列出从 Neo4j 获取的信息 (包含在 `PaperDetailResponse` 中): `tasks`, `datasets`, `methods`。这些应作为可点击的链接（如果目标实体也有详情页或搜索功能）。
        -   **展示图邻接信息:** 提供一个区域（可能是 Tab 或独立板块）用于展示从 `GET /api/v1/graph/papers/{pwc_id}/graph` 获取的 `GraphData`。**初期**可以是一个简单的列表，展示邻接 `Node` 的 `label` 和 `type` (如 'Task: Image Classification', 'Dataset: ImageNet')。
    -   **模型详情页面 (`HFModelDetail`):**
        -   展示从 `GET /api/v1/graph/models/{model_id}` 获取的结构化元数据: `model_id`, `author`, `sha`, `last_modified`, `tags`, `pipeline_tag`, `downloads`, `likes`, `library_name`, `created_at`, `updated_at`。
        -   **(可选/未来):** 调用 `GET /api/v1/graph/related` 端点获取与该模型相关的论文、数据集等，并展示。

**3. 简化架构**

采用 **客户端渲染 (Client-Side Rendering, CSR) 的单页应用 (Single Page Application, SPA)** 架构。开发环境通过 **Podman Compose** 统一管理。

```mermaid
graph TD
    subgraph "开发环境 (Podman Compose Managed)"
        direction LR
        Browser[开发浏览器] <--> FE_DevServer[前端开发服务器容器<br>(Vite on Node.js)]
        FE_DevServer -- API Proxy (Handles /api/*) --> BE_API[后端 FastAPI 容器<br>(Provides Search, Graph, Details)]
        BE_API <--> PG_DB[PostgreSQL<br>(Keyword Search, Details)]
        BE_API <--> N4J_DB[Neo4j<br>(Graph Queries, Relations)]
        BE_API -- Uses --> FAISS_Idx[Faiss Index<br>(Semantic Search)] # Faiss managed by Backend
    end

    subgraph "前端应用内部 (Browser)"
        direction TB
        UserInteraction[用户交互<br>(输入查询, 选择模式/过滤/排序)] --> AppState[应用状态<br>(UI State + URL State + TanStack Query Cache)]
        AppState -- Drives --> APIParams[API 参数构造]
        APIParams -- Triggers --> API_Client[API 请求层<br>(TanStack Query)]
        API_Client -- Makes Request --> FE_DevServer # Via Proxy
        FE_DevServer -- Returns Data --> API_Client
        API_Client -- Updates --> AppState
        AppState -- Renders --> ReactUI[React UI<br>(Components, Pages, Routing)]
        ReactUI -- Includes --> GraphViz[图信息展示<br>(List / Basic Viz)]
    end

    User[开发者] -- Edits Code --> LocalFS[本地文件系统<br>(挂载到 FE 容器)]

    style FE_DevServer fill:#EBF5FB,stroke:#333,stroke-width:2px
    style BE_API fill:#FEF9E7,stroke:#333,stroke-width:2px
    style PG_DB fill:#D6EAF8,stroke:#333,stroke-width:1px
    style N4J_DB fill:#D5F5E3,stroke:#333,stroke-width:1px
    style FAISS_Idx fill:#FADBD8,stroke:#333,stroke-width:1px
    style AppState fill:#FDEDEC,stroke:#333,stroke-width:1px
    style API_Client fill:#D5F5E3,stroke:#333,stroke-width:1px
    style ReactUI fill:#E8DAEF,stroke:#333,stroke-width:1px
```

**关键架构决策:**

-   **SPA & CSR:** 保持流畅体验和部署简单性。
-   **组件驱动:** UI 由可复用的 React 组件构成 (搜索栏, 过滤器面板, 结果列表, 详情卡片, 关联列表等)。
-   **API 客户端层 (TanStack Query):** 核心工具，用于管理所有后端数据获取、缓存、状态同步和参数化请求。
-   **状态管理:**
    -   **服务器状态:** 由 TanStack Query 主导。
    -   **UI / 搜索参数状态:** 使用 React 内建状态 (`useState`, `useReducer`) 和 Context API。考虑使用 URL 查询参数 (`useSearchParams` from React Router) 来同步搜索、过滤、排序和分页状态，以支持分享和浏览器历史。
-   **客户端路由 (React Router):** 管理页面导航和 URL 状态同步。
-   **容器化开发环境 (Podman Compose):** 确保环境一致性和便捷性。

**4. 推荐技术选型 (务实且高效)**

| 类别                    | 推荐技术                          | 理由                                                                               |
| :---------------------- | :-------------------------------- | :--------------------------------------------------------------------------------- |
| **核心框架** | **React (v18+)** | 强大的生态、组件化模型、社区支持广泛、与强类型 (TS) 结合良好。                       |
| **构建工具** | **Vite** | 极快的冷启动和热更新速度，现代化的构建标准，配置简单。                               |
| **编程语言** | **TypeScript** | **强制要求。** 提供静态类型检查，减少运行时错误，提高代码可维护性。                    |
| **路由管理** | **React Router (v6+)** | React 应用的事实标准，支持 URL 参数同步状态。                                        |
| **数据请求/状态** | **TanStack Query (React Query)** | **强烈推荐。** 核心库，管理数据获取、缓存、同步、错误处理、参数化请求。                |
| **HTTP 客户端** | **Axios** 或 **Fetch API** | TanStack Query 可与两者配合。                                                        |
| **UI 样式** | **Tailwind CSS** | **推荐。** Utility-first，快速构建定制化、一致的 UI。                                |
| **UI 组件库** | **无 (初期) / Headless UI (可选)** | **大道至简。** 先用 Tailwind 构建。复杂交互组件（如下拉菜单、日期选择器）可考虑 Radix UI 或 Headless UI。 |
| **图可视化 (可选/未来)** | **react-flow / vis.js / d3.js** | 用于在详情页展示邻接图。初期可仅用列表，后续按需引入。                               |
| **状态管理 (补充)** | **React Context/URL Params** | **MVP 阶段足够。** 优先使用 React 内建能力和 URL 同步搜索参数。                      |
| **代码检查** | **ESLint** | **强制要求。** 识别代码风格问题和潜在错误。                                          |
| **代码格式化** | **Prettier** | **强制要求。** 自动统一代码格式。                                                  |
| **测试框架** | **Vitest** | 与 Vite 集成良好，API 类似 Jest，速度快。                                          |
| **组件测试** | **React Testing Library (RTL)** | **推荐。** 鼓励面向用户交互方式测试组件，覆盖不同搜索参数/状态。                     |
| **包管理器** | **pnpm** (推荐) 或 yarn             | 更高效的依赖管理。                                                                 |
| **Node 版本管理** | **nvm** (或 volta, asdf)          | 管理本地 Node.js 版本 (容器化环境已指定)。                                           |

**5. 目录结构 (建议)**

```
aigraphx-frontend/
├── public/
├── src/
│   ├── App.tsx
│   ├── main.tsx
│   ├── assets/
│   ├── components/
│   │   ├── common/       # Button, Input, Spinner, Pagination, DatePicker, MultiSelect...
│   │   ├── layout/       # Header, Footer, FilterSidebar...
│   │   ├── search/       # SearchBar, SearchModeToggle, FilterPanel(包含论文/模型过滤器), SortDropdown, ResultList, ResultItem(区分Paper/Model)...
│   │   ├── details/      # PaperDetailCard, ModelDetailCard, MetadataSection, RelatedEntitiesList...
│   │   └── graph/        # (可选/未来) GraphVisualizationComponent...
│   ├── config/
│   ├── constants/      # 如搜索类型、排序选项常量
│   ├── hooks/          # useSearchParameters (管理 URL 同步), useDebounce...
│   ├── pages/          # SearchPage, ModelDetailPage, PaperDetailPage, NotFoundPage...
│   ├── services/
│   │   ├── apiClient.ts
│   │   └── apiQueries.ts # 定义各种 useQuery/useMutation hooks (e.g., useSemanticPaperSearch, useKeywordModelSearch, useHybridPaperSearch, usePaperDetails, useModelDetails, usePaperGraph, useRelatedEntities...)
│   ├── store/          # (可选) Context providers/hooks
│   ├── styles/
│   ├── types/
│   │   └── api.ts      # !!! 极其重要: 必须与后端 /docs 严格同步的类型 (SearchResultItem, HFSearchResultItem, PaginatedPaperSearchResult, PaginatedHFModelSearchResult, SearchFilterModel, PaperDetailResponse, HFModelDetail, GraphData, Node, Relationship 等), 强烈推荐使用 openapi-typescript 自动生成 !!!
│   ├── utils/
│   └── router/         # 路由配置, 可能包含 URL 参数解析逻辑
├── tests/
│   └── (结构镜像 src/)   # 重点测试 SearchPage(带参数), DetailPage, apiQueries mocks
├── .env*
├── .eslintrc.cjs
├── .gitignore
├── .prettierrc.json
├── index.html
├── package.json
├── pnpm-lock.yaml
├── tailwind.config.js
├── tsconfig*.json
├── vite.config.ts
└── Containerfile       # (保持不变)
```

**6. API 交互策略**

* **权威来源:** **必须**以后端运行实例的 **Swagger UI (`/docs`)** 作为所有 API 端点 (如 `/api/v1/search/...`, `/api/v1/graph/...`)、请求参数 (`q`, `target`, `search_type`, `page`, `page_size`, `published_after`, `published_before`, `filter_area`, `sort_by`, `sort_order` 等)、响应结构和可用选项的**唯一权威参考**。
* **封装:** 在 `src/services/apiQueries.ts` 中使用 TanStack Query 的 `useQuery`, `useInfiniteQuery` (适合分页) 和 `useMutation` 封装所有 API 调用。函数命名应清晰反映操作 (e.g., `useSemanticPaperSearch`).
* **参数化:**
    * 前端 UI 状态（搜索词、选择的搜索模式/目标、激活的过滤器、排序选项、当前页码）**必须**正确地映射到 API 请求参数。
    * 推荐使用自定义 Hook (`useSearchParameters`) 结合 React Router 的 `useSearchParams` 来管理这些状态并与 URL 同步。
    * 考虑对用户输入（如搜索词）进行 debounce 处理，避免频繁触发 API 请求。
* **类型安全:** **必须**在 `src/types/api.ts` 中定义与 `/docs` 完全一致的 TypeScript 类型（包括 `SearchResultItem`, `HFSearchResultItem`, `PaginatedPaperSearchResult`, `PaginatedHFModelSearchResult`, `SearchFilterModel`, `PaperDetailResponse`, `HFModelDetail`, `GraphData`, `Node`, `Relationship` 等）。**强烈推荐**使用 `openapi-typescript` 或类似工具基于后端的 `/openapi.json` **自动生成**这些类型，并在后端 API 变更时重新生成。
* **错误处理:** 利用 TanStack Query 的 `isError`, `error` 状态在 UI 中提供清晰、友好的错误反馈。
* **配置与代理:**
    * **开发环境 (容器化):** 使用 `vite.config.ts` 中的代理将 `/api/*` 请求转发到后端容器 (e.g., `http://backend:8000`)。API 调用使用相对路径 (`/api/v1/...`)。
    * **生产环境:** API 基础 URL 通过构建时环境变量 `VITE_API_BASE_URL` 注入 (`import.meta.env.VITE_API_BASE_URL`)。

**7. 状态管理策略 (强化)**

* **服务器状态 (数据缓存):** **核心由 TanStack Query 管理**。利用其缓存能力优化性能，减少不必要的请求。
* **URL 状态 (搜索/过滤参数):** **强烈推荐**使用 React Router 的 `useSearchParams` Hook 将当前的搜索词 (`q`), 搜索模式 (`search_type`), 目标 (`target`), 过滤器 (`published_after`, `published_before`, `filter_area`), 排序选项 (`sort_by`, `sort_order`) 和页码 (`page`) 同步到 URL 查询参数。这使得状态可分享、可收藏，并支持浏览器前进/后退。可以封装一个自定义 Hook (`useSearchParameters`) 来简化读写 URL 状态并将其提供给组件。
* **全局 UI 状态 (少量):** 如有必要（例如，全局通知、用户偏好），使用 React Context API。
* **局部 UI 状态:** 组件内部状态（如下拉菜单是否打开）使用 `useState`。

**8. UI/UX 初步概念 (扩展)**

* **搜索页面:**
    * **顶部/显眼位置:** 搜索输入框 (`q`), **搜索模式选择器 (Tabs/Radio - `search_type`)**, **目标实体选择器 (Dropdown/Tabs - `target`)**, 触发按钮。
    * **侧边栏/可折叠面板:** **过滤器区域**，包含**论文过滤器** (日期范围 `published_after`/`published_before`, 领域 `filter_area`)。提供"清除筛选"按钮。(模型过滤器待后端 `/docs` 确认)。
    * **结果列表上方:** 显示当前结果数量 (`total` from pagination), **排序方式下拉菜单 (`sort_by`, `sort_order` - 根据 target 和 search_type 显示可用选项)**, **分页控件 (`page`, `page_size`)**。
    * **结果列表项:** 显示核心信息 (根据 `SearchResultItem` / `HFSearchResultItem` 字段)，突出显示**匹配关键词 (若可能)** 或 **相关性分数 (`score`)** (注意 score 可能为 null)。
* **详情页面:**
    * **主区域 (论文 - `PaperDetailResponse`):** 清晰展示核心元数据（`title`, `abstract`, `authors`, `published_date`, `area`, `arxiv_id`, `url_abs`, `url_pdf`, `frameworks`, `number_of_stars` 等）。
    * **主区域 (模型 - `HFModelDetail`):** 清晰展示核心元数据 (`model_id`, `author`, `pipeline_tag`, `library_name`, `tags`, `likes`, `downloads`, `last_modified` 等)。
    * **侧边栏/下方区域:**
        - **关联实体列表 (论文):** 分组展示 `tasks`, `datasets`, `methods`，列表项应为**可点击链接**。
        - **图谱邻接视图 (论文 - `GraphData`):** 一个专门的区域（可能是 Tab），**初期**展示关联 `Node` 的简单列表 (e.g., "Task: Text Classification (ID: task_123)", "Dataset: GLUE (ID: dataset_abc)")。**后期**可升级为交互式图表。
* **整体:** 保持界面干净，提供清晰的视觉层次。加载、错误、空状态反馈要明确。响应式设计确保在桌面端良好可用。

**9. 开发环境设置 (容器化优先)**

* (保持 v1.1 中的内容不变，重点是 Podman Compose 配置和启动流程)
    * 确保 `compose.yml` 中的 `frontend` 服务配置正确（context, volumes, port, network）。
    * 确保 `vite.config.ts` 中的 `server.proxy` 配置正确指向后端服务名 (e.g., `target: 'http://backend:8000'`)。

**10. 代码规范与质量 (!!! 关键 !!!)**

* (保持 v1.1 中的严格要求不变: 强制 TS, ESLint, Prettier, 命名规范, 类型提示, Props 定义, 显式导入, 注释)

**11. 测试策略 (适配扩展功能)**

* **核心:** 继续使用 **RTL + Vitest + MSW**。
* **扩展测试范围:**
    * **组件测试:** 测试过滤器组件（日期选择、文本输入 `filter_area`）、排序下拉菜单、分页控件、搜索模式/目标切换器。
    * **集成测试 (页面级):**
        * 模拟用户选择不同**搜索模式**和**目标**，验证传递给 API 的 `search_type`, `target` 及正确的端点被调用。
        * 模拟用户应用**多种过滤器** (`published_after`, `filter_area`) 和**排序** (`sort_by`, `sort_order`)，验证 API 请求参数的正确性及 UI 的相应更新。
        * 模拟用户**分页**操作 (`page`, `page_size`)。
        * 测试详情页**元数据**、**关联实体列表**和**基础图信息列表**的正确渲染（基于 Mock 数据，使用 `PaperDetailResponse`, `HFModelDetail`, `GraphData` 结构）。
    * **Mocking:** 使用 MSW 模拟后端 API，覆盖不同的搜索结果 (`PaginatedPaperSearchResult`, `PaginatedHFModelSearchResult`)、详情数据 (`PaperDetailResponse`, `HFModelDetail`)、图数据 (`GraphData`) 以及错误情况。
* **CI 强制:** (保持 v1.1 要求不变) `tsc --noEmit`, `eslint .`, `prettier --check .`, `vitest run` 必须通过。

**12. 构建与部署**

* (保持 v1.1 中的内容不变: `pnpm build`, 静态托管部署, 构建时环境变量注入, CI/CD)

**13. 未来增强功能 (修订)**

* **交互式图谱可视化:** 在详情页提供可交互、可探索的图谱视图（缩放、平移、节点点击、邻居展开）。
* **跨实体链接:** 在 UI 中实现更丰富的交叉链接，例如从论文详情页直接跳转到其提及的任务的详情页（如果任务也有详情页）。
* **可视化定制:** 允许用户自定义图可视化布局或过滤显示的节点/关系类型。
* **用户账户与个性化:** 保存搜索历史、收藏夹、个性化推荐。
* **性能优化:** 对大规模数据渲染和复杂图可视化进行性能分析和优化。
* **端到端测试:** 使用 Playwright/Cypress 覆盖核心用户流程（多模式搜索+过滤+排序+查看详情+查看关联）。

**14. 结论**

这份更新后的前端设计文档 (v1.1 - Expanded Scope) 旨在指导开发一个能充分利用 AIGraphX 后端强大功能的用户界面。通过实现**多模式搜索、高级参数化查询、详情关联探索**，并继续采用 **React + Vite + TypeScript + Tailwind CSS + TanStack Query** 的技术栈和**容器化开发环境**，我们可以构建一个功能强大、用户体验良好且可维护的前端应用。**严格遵守代码规范、测试策略，并始终以后端 `/docs` 作为 API 的最终事实依据**，是项目成功的关键。

---

**下一步建议:**

* (保持 v1.1 中的建议不变，重点是设置和启动容器化环境)

* **优先实现:** 搜索框 (`q`), 搜索模式切换 (`search_type`), 目标选择 (`target`), 基础结果展示 (区分论文/模型), 分页 (`page`/`page_size`), 排序 (`sort_by`/`sort_order` - 根据 `/docs` 动态选项), 过滤 (论文日期 `published_after`/`published_before`, 领域 `filter_area`), 论文详情页元数据 (`PaperDetailResponse`), 模型详情页元数据 (`HFModelDetail`), 论文详情页关联实体列表 (`tasks`, `datasets`, `methods`)。基础图信息列表展示可稍后。

使用 openapi-typescript 或类似工具，基于后端 /openapi.json 初始化 src/types/api.ts 文件。

---
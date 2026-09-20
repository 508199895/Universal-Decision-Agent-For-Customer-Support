# SQLAlchemy ORM 的通用使用流程

SQLAlchemy ORM 的通用流程是：用 Python 类描述数据库表，通过 Engine 建立数据库连接，通过 Session 查询和修改数据，最后提交事务。

## 执行顺序

| 执行顺序 | 阶段 | 核心概念 | 典型代码 | 作用 |
|---:|---|---|---|---|
| 1 | 导入组件 | SQLAlchemy API | `from sqlalchemy import create_engine, Column, String` | 导入连接、字段类型和 ORM 所需组件 |
| 2 | 创建模型基类 | `DeclarativeBase` / `Base` | `Base = declarative_base()` | 建立所有 ORM 模型共同继承的基类，并收集表结构元数据 |
| 3 | 定义 ORM 模型 | Model | `class User(Base): ...` | 用 Python 类表示数据库表 |
| 4 | 指定表名 | `__tablename__` | `__tablename__ = "users"` | 建立 ORM 类与数据库表的映射 |
| 5 | 定义字段 | `Column` | `name = Column(String)` | 定义列名、数据类型和约束 |
| 6 | 定义约束和关联 | PK、FK、Unique、Relationship | `ForeignKey(...)`、`relationship(...)` | 定义主键、外键、唯一性和对象关系 |
| 7 | 创建数据库引擎 | `Engine` | `create_engine("sqlite:///app.db")` | 配置数据库地址、驱动和连接方式 |
| 8 | 创建数据库表 | Metadata / DDL | `Base.metadata.create_all(engine)` | 根据 ORM 模型执行必要的 `CREATE TABLE` |
| 9 | 创建会话工厂 | `sessionmaker` | `SessionLocal = sessionmaker(bind=engine)` | 定义产生数据库会话的方法 |
| 10 | 创建会话 | `Session` | `session = SessionLocal()` | 建立一次数据库操作上下文 |
| 11 | 执行业务操作 | CRUD | `add()`、`select()`、`delete()` | 新增、查询、修改或删除记录 |
| 12 | 提交事务 | Transaction | `session.commit()` | 将新增、修改或删除永久写入数据库 |
| 13 | 异常处理 | Rollback | `session.rollback()` | 操作失败时撤销当前事务 |
| 14 | 关闭会话 | Connection lifecycle | `session.close()` | 释放数据库连接等资源 |

## 1. 定义 ORM 模型

现代 SQLAlchemy 通常这样定义基类：

```python
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass
```

然后定义模型：

```python
from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column


class User(Base):
    __tablename__ = "users"

    user_id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
```

| Python ORM | 数据库 |
|---|---|
| `User` 类 | `users` 表 |
| `User.user_id` 属性 | `user_id` 字段 |
| `User` 对象 | 一条用户记录 |
| `Base.metadata` | 全部 ORM 表结构信息 |

## 2. 常用字段约束

| ORM 参数或对象 | 数据库含义 | 示例 |
|---|---|---|
| `primary_key=True` | 主键 | `id = Column(String, primary_key=True)` |
| `nullable=False` | 不允许为空 | `name = Column(String, nullable=False)` |
| `unique=True` | 值不能重复 | `email = Column(String, unique=True)` |
| `index=True` | 创建普通索引 | `user_id = Column(String, index=True)` |
| `default=...` | Python/SQLAlchemy 侧默认值 | `status = Column(String, default="open")` |
| `server_default=...` | 数据库侧默认值 | `server_default="open"` |
| `ForeignKey(...)` | 外键 | `ForeignKey("users.user_id")` |
| `Text` | 长文本类型 | `content = Column(Text)` |
| `DateTime` | 日期时间类型 | `created_at = Column(DateTime)` |

## 3. 创建 Engine

```python
from sqlalchemy import create_engine

engine = create_engine("sqlite:///app.db")
```

`Engine` 是数据库连接配置和连接管理的入口，不代表某一条业务记录。

| 数据库 | URL 示例 |
|---|---|
| SQLite | `sqlite:///app.db` |
| PostgreSQL | `postgresql+psycopg://user:password@host/dbname` |
| MySQL | `mysql+pymysql://user:password@host/dbname` |

## 4. 创建表

```python
Base.metadata.create_all(bind=engine)
```

执行逻辑：

```text
读取所有 Base 子类
    ↓
收集表名、字段和约束
    ↓
检查数据库中是否已有对应表
    ↓
没有则执行 CREATE TABLE
    ↓
已有则跳过
```

`create_all()` 一般不会删除已有表、删除已有数据、自动修改已有字段或完成数据库版本迁移。修改现有表结构通常应使用 Alembic 等迁移工具。

## 5. 创建并使用 Session

```python
from sqlalchemy.orm import sessionmaker

SessionLocal = sessionmaker(bind=engine)

with SessionLocal() as session:
    ...
```

| 对象 | 作用 |
|---|---|
| `Engine` | 管理如何连接数据库 |
| `SessionLocal` | 创建 Session 的工厂 |
| `Session` | 执行一次或一组数据库操作 |
| Transaction | 保证一组写操作全部成功或全部撤销 |

## 6. CRUD 操作

### 新增

```python
user = User(user_id="u001", name="Alice")
session.add(user)
session.commit()
```

```text
创建 Python 对象
→ session.add()
→ 进入待写入状态
→ session.commit()
→ 执行 INSERT
```

### 查询

SQLAlchemy 2.x 推荐使用 `select()`：

```python
from sqlalchemy import select

stmt = select(User).where(User.user_id == "u001")
user = session.scalar(stmt)
```

对应 SQL 类似：

```sql
SELECT *
FROM users
WHERE user_id = 'u001';
```

### 修改

```python
user.name = "Alice Smith"
session.commit()
```

```text
查询 ORM 对象
→ 修改 Python 属性
→ Session 检测到变化
→ commit()
→ 执行 UPDATE
```

通常不需要再次调用 `session.add(user)`。

### 删除

```python
session.delete(user)
session.commit()
```

提交后会执行对应的 `DELETE`。

## 7. 外键与 relationship

数据库外键：

```python
class Ticket(Base):
    __tablename__ = "tickets"

    user_id = Column(
        String,
        ForeignKey("users.user_id"),
        nullable=False,
    )
```

ORM 对象关系：

```python
class User(Base):
    tickets = relationship("Ticket", back_populates="user")


class Ticket(Base):
    user = relationship("User", back_populates="tickets")
```

| 概念 | 工作层级 | 作用 |
|---|---|---|
| `ForeignKey` | 数据库 | 保证关联字段指向有效记录 |
| `relationship` | Python ORM | 方便通过对象访问关联数据 |

例如，`user.tickets` 可以取得该用户关联的工单，而 `ticket.user` 可以取得工单所属用户。

## 8. 推荐的事务模板

```python
SessionLocal = sessionmaker(bind=engine)

with SessionLocal() as session:
    try:
        user = User(user_id="u001", name="Alice")
        session.add(user)
        session.commit()
    except Exception:
        session.rollback()
        raise
```

完整执行顺序：

```text
定义 Base
  ↓
定义 ORM 模型
  ↓
创建 Engine
  ↓
create_all() 或执行数据库迁移
  ↓
创建 Session 工厂
  ↓
打开 Session
  ↓
执行查询或数据修改
  ↓
写操作 commit()
  ├── 成功 → 完成
  └── 失败 → rollback()
  ↓
关闭 Session
```

## 9. 在本项目中的对应实现

| 通用步骤 | 本项目实现 |
|---|---|
| 定义 Base 和模型 | `data/models/cultpass.py`、`data/models/udahub.py`、`mcp_services/memory/server.py` |
| 创建 Engine | 两个数据库初始化 Notebook 及三个 MCP 服务 |
| 建表 | Notebook 中的 `Base.metadata.create_all()`；记忆服务启动时的 `create_all()` |
| 创建 Session | `utils.get_session()` 或 MCP 服务中的 `SessionLocal` |
| 查询 | Knowledge、Account、Memory MCP 工具 |
| 写入 | 初始化 Notebook，以及 `memory_write()` |
| 提交或回滚 | `get_session()` 上下文管理器或显式调用 `commit()`、`rollback()` |


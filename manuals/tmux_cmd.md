## tmux使用

### 基础指令

启动一个默认session（默认名称）：
``` bash
tmux
```

新建一个自定义命名的session：

```bash
tmux new -s mysession
```

创建一个新session或者连接到一个已存在但是重合名字的session：
```bash
tmux new-session -A -s mysession
```
>这样可以避免新建重复session

### 一些其他建议

* 取消链接到一个session，按下`Ctrl + b`，再按下`d`.

* 删除一个session，按下`Ctrl + b`, 再按下`&`.

* 查看当前所有session

```bash
tmux ls
```

* 重连到一个已存在的session：
```bash
tmux attach-session -t mysession
```

* 删除一个session
```bash
tmux kill-session -t session-name
```

* 关闭所有会话：
```bash
tmux kill-server
```
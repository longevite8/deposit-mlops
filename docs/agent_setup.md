# ClearML Agent Setup & Configuration

## Overview

The **ClearML Agent** executes pipeline tasks on remote machines. Configure agents to run on different hardware resources (CPU queue, GPU queue, services queue).

---

## Architecture

```text
ClearML Server
    ↓
Task Queue (e.g., "cpu_queue")
    ↓
ClearML Agent (listens to queue)
    ↓
Dequeues & Executes Task
    ↓
Reports Results to Server
```

---

## Agent Setup

### Step 1: Install ClearML Agent

```bash
pip install clearml-agent
```

### Step 2: Configure Credentials

```bash
clearml-agent init
```

**Prompts:**

1. ClearML Web URL: `http://localhost:8080`
2. API Access Key: (from ClearML settings)
3. API Secret Key: (from ClearML settings)
4. File server URL: `http://localhost:8081`

**Output:** Saves to `~/.clearml/clearml.conf`

### Step 3: Verify Configuration

```bash
clearml-agent info
```

Output shows:

* ClearML version
* Python version
* Available GPU/CPU
* Queues configured

---

## Start Agent

### Basic CPU Agent

Run tasks on CPU queue:

```bash
clearml-agent daemon --queue cpu_queue
```

This:

* Connects to `cpu_queue`
* Waits for tasks
* Executes immediately when task appears
* Reports progress to server
* Never exits (daemon mode)

### Multiple CPU Worker Agents

Scale by running multiple agents:

```bash
# Terminal 1
clearml-agent daemon --queue cpu_queue --name worker-1

# Terminal 2
clearml-agent daemon --queue cpu_queue --name worker-2

# Terminal 3
clearml-agent daemon --queue cpu_queue --name worker-3
```

**Benefits:**

* Parallel task execution
* Load balancing
* Fault tolerance (if one dies, others continue)

### GPU Agent

For GPU-accelerated tasks:

```bash
clearml-agent daemon --queue gpu_queue --gpu-index 0,1
```

Allocates GPUs 0 and 1 to this agent.

### Services Queue Agent

For auto-retraining (background trigger):

```bash
clearml-agent daemon --queue services --workers 2
```

Runs on services queue with 2 worker threads.

---

## Docker Agent

Run agent inside Docker container:

### Prerequisites

```bash
docker pull nvidia/cuda:11.8.0-runtime-ubuntu22.04
```

### Start Docker Agent

```bash
clearml-agent daemon \
  --queue cpu_queue \
  --docker nvidia/cuda:11.8.0-runtime-ubuntu22.04 \
  --docker-bash-setup "apt-get update && apt-get install -y python3 python3-pip"
```

**Benefits:**

* Isolated environment
* Consistent dependencies
* Easy cleanup (just stop container)

### Docker Compose Setup

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  agent:
    image: nvidia/cuda:11.8.0-runtime-ubuntu22.04
    environment:
      CLEARML_API_HOST: http://clearml-server:8008
      CLEARML_WEB_HOST: http://localhost:8080
      CLEARML_API_ACCESS_KEY: ${CLEARML_API_ACCESS_KEY}
      CLEARML_API_SECRET_KEY: ${CLEARML_API_SECRET_KEY}
    command: >
      bash -c "
        apt-get update &&
        apt-get install -y python3 python3-pip &&
        pip install clearml &&
        clearml-agent daemon --queue cpu_queue
      "
    depends_on:
      - clearml-server
```

Run:

```bash
docker-compose up -d
```

---

## Queue Configuration

### Project Queues

The project uses:

* **cpu_queue** - Training & inference (regular CPU tasks)
* **services** - Auto-retraining (background service)

### Create New Queue

```bash
clearml-agent daemon --queue new_queue_name
```

Or via ClearML Web UI:

1. Workers tab
2. Queues
3. * New Queue
4. Enter queue name

---

## Agent Monitoring

### View Active Agents

```bash
clearml-agent status
```

Output:

```
Agents:
  agent-1
    Queue: cpu_queue
    Status: Running
    Tasks executed: 42
    Uptime: 3 days
```

### View Queue Status

Via ClearML Web UI:

1. Workers → Queues
2. Select queue (e.g., cpu_queue)
3. View pending tasks and workers

### Monitor Task Execution

Via ClearML Web UI:

1. Projects → Pipelines
2. Click pipeline task
3. View live output stream
4. Check resource usage (CPU, memory, GPU)

---

## Agent Configuration File

Location: `~/.clearml/clearml.conf`

Key sections:

```ini
[agent]
queue_name = cpu_queue
log_level = INFO
worker_threads = 1  # Tasks executed in parallel
max_concurrent_tasks_per_worker = 1

[agent.git]
git_user = your-github-username
git_password = your-github-token

[agent.docker]
docker_user = your-docker-user
docker_password = your-docker-password
```

### Reload Configuration

```bash
clearml-agent daemon --queue cpu_queue --reload
```

---

## Common Issues & Solutions

### Issue: Agent Won't Connect

**Symptom:** `Connection refused` error

**Solution:**

1. Verify ClearML server is running
2. Check `~/.clearml/clearml.conf` hosts
3. Test connectivity: `curl http://localhost:8080`
4. Restart agent: `clearml-agent daemon --queue cpu_queue`

### Issue: Tasks Timeout

**Symptom:** Task runs indefinitely, gets killed after 24 hours

**Solution:**

* Set task timeout in code:

  ```python
  task.set_time_limit(minutes=60)  # 1 hour timeout
  ```

* Or increase ClearML server timeout:
  Edit `~/.clearml/clearml.conf`:

  ```ini
  [agent]
  task_execution_timeout = 86400  # seconds
  ```

### Issue: Wrong Python Version

**Symptom:** Tasks fail with module import errors

**Solution:**

```bash
# Check agent's Python
clearml-agent info | grep Python

# Use specific Python
clearml-agent daemon --queue cpu_queue --python /usr/bin/python3.11
```

### Issue: GPU Not Detected

**Symptom:** `torch.cuda.is_available() = False` in task

**Solution:**

```bash
# Check GPU available to agent
clearml-agent info | grep GPU

# Specify GPU explicitly
clearml-agent daemon --queue gpu_queue --gpu-index 0
```

### Issue: Disk Space Full

**Symptom:** Tasks fail randomly, agent offline

**Solution:**

```bash
# Cleanup ClearML cache
rm -rf ~/.clearml/cache/*

# Check disk usage
df -h

# Cleanup old Docker images
docker image prune -a --force
```

---

## Performance Tuning

### Increase Worker Threads

Edit `~/.clearml/clearml.conf`:

```ini
[agent]
worker_threads = 4  # Default: 1
```

**Effect:** Multiple tasks execute in parallel (requires multi-core CPU)

### Batch Processing

For data-heavy tasks, batch into fewer, larger tasks:

```python
# Instead of: 100 small tasks
for i in range(100):
    queue_task(small_data[i])

# Do: 10 large batches
for batch_idx in range(10):
    queue_task(small_data[batch_idx*10:(batch_idx+1)*10])
```

### Network Optimization

For slow connections:

```bash
clearml-agent daemon \
  --queue cpu_queue \
  --git-clone-depth 1  # Shallow clone
  --disable-lock-file   # Skip lock file checks
```

---

## Security Best Practices

### 1. Use Environment Variables

Never hardcode credentials:

```bash
export CLEARML_API_ACCESS_KEY=your-key
export CLEARML_API_SECRET_KEY=your-secret
export ALERT_SMTP_PASSWORD=your-password

clearml-agent daemon --queue cpu_queue
```

### 2. Restrict Queue Access

Limit who can enqueue tasks to a queue:

Edit `~/.clearml/clearml.conf`:

```ini
[agent]
authorized_user_ids = ["user_id_1", "user_id_2"]
```

### 3. Monitor Agent Logs

```bash
tail -f ~/.clearml/logs/agent.log
```

Look for:

* Unauthorized access attempts
* Failed authentications
* Unusual task activity

### 4. Regularly Update

```bash
pip install --upgrade clearml clearml-agent
```

---

## Integration with Pipelines

### Training Pipeline

Queue: `cpu_queue`

```python
task.execute_remotely(queue_name="cpu_queue")
```

Or in pipeline:

```python
pipeline.add_step(..., queue="cpu_queue")
```

### Production Pipeline

Queue: `cpu_queue`

```python
pipeline.add_step(..., queue="cpu_queue")
```

### Auto-Retraining

Queue: `services`

```python
# In auto_retraining.py
pipeline.start(queue_name=SERVICES_QUEUE)
```

---

## Local vs Remote Execution

### Local Execution (for testing)

```bash
# Task runs on your machine
python tasks/train_model.py
```

### Remote Execution (via agent)

```python
task = Task.init(...)
task.execute_remotely(queue_name="cpu_queue")
# Task enqueued, executed by agent
```

### Hybrid Approach

* Local: debug & develop
* Remote: production pipelines

```bash
# Terminal 1: Debug locally
python tasks/train_model.py

# Terminal 2: Agent running
clearml-agent daemon --queue cpu_queue

# Terminal 3: Enqueue via pipeline
python pipelines/trainning_pipeline.py
```

---

## Troubleshooting Checklist

* [ ] `clearml-agent info` shows correct server
* [ ] Agent can reach ClearML server (`curl http://localhost:8080`)
* [ ] Queue name matches pipeline configuration
* [ ] Python version matches requirements
* [ ] GPU enabled if GPU task
* [ ] Disk space available (>5GB recommended)
* [ ] Agent logs show `Listening to queue: cpu_queue`
* [ ] Tasks appear in queue but don't execute?
  * Check agent log for errors
  * Verify worker threads > 0
  * Check resource limits

---

## References

* [ClearML Agent Documentation](https://clear.ml/docs/latest/docs/references/cli/cli_agent/)
* [Queues & Workers](https://clear.ml/docs/latest/docs/fundamentals/agents_and_queues/)
* [Docker Agent](https://clear.ml/docs/latest/docs/references/sdk/agent_docker/)

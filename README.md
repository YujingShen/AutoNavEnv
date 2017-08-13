# AutoNavEnv

## Core API

```python
from AutoNavEnv.env import AutoNavEnv
env = AutoNavEnv()
```

* env.make(folder/path/contains/images)
* observation = env.reset()  
* observation, reward, done, info = env.step(...)
* env.render() 

###### PS: no need to call render during training

## Helper Funcs

* env.num\_of_operations
* env.num\_of_moves
* env.num\_of_collisions

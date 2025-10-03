from agent.envs.calc_env import CalcEnv
from agent.base.env import Env
def get_env(env_config)->Env:
    if env_config.get("env_name") == "calc":
        return CalcEnv(env_config)
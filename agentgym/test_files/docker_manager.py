import docker
import time
from typing import Dict, Any, Optional
from docker.models.containers import Container


class DockerManager:
    def __init__(self):
        self.client = docker.from_env()
        self.operation_lock = False
        self.lock_timeout = 30.0
        self.retry_delay = 0.1

    def _acquire_lock(self):
        start_time = time.time()
        while self.operation_lock:
            if time.time() - start_time > self.lock_timeout:
                raise Exception("Operation timeout: failed to acquire lock")
            time.sleep(self.retry_delay)
        self.operation_lock = True

    def _release_lock(self):
        self.operation_lock = False

    def create_container(self, image_name: str, container_name: Optional[str] = None) -> Container:
        self._acquire_lock()
        try:
            container = self.client.containers.create(
                image_name,
                name=container_name,
                auto_remove=False,
                restart_policy={"Name": "no"}
            )
            return container
        finally:
            self._release_lock()

    def start_container(self, container_id: str):
        self._acquire_lock()
        try:
            container = self.client.containers.get(container_id)
            container.start()
        finally:
            self._release_lock()

    def remove_container(self, container_id: str):
        self._acquire_lock()
        try:
            container = self.client.containers.get(container_id)
            container.remove(force=True)
        finally:
            self._release_lock()

    def get_container_info(self, container_id: str) -> Dict[str, Any]:
        container = self.client.containers.get(container_id)
        return container.attrs

    def list_containers(self, image_name: str = "train-env:latest") -> list:
        all_containers = self.client.containers.list(all=True)
        if ':' in image_name:
            filtered = [c for c in all_containers if c.image.tags and image_name in c.image.tags]
        else:
            filtered = [c for c in all_containers if c.image.tags and 
                       any(tag.startswith(f"{image_name}:") for tag in c.image.tags)]
        return filtered

    def get_container_ip(self, container_id: str) -> str:
        container = self.client.containers.get(container_id)
        network_settings = container.attrs['NetworkSettings']
        ip_address = network_settings.get('IPAddress')
        if not ip_address:
            raise Exception(f"No IP address found for container {container_id}")
        return ip_address

    def get_container_tool_list(self, container_id: str) -> Dict[str, Any]:
        import requests
        
        container_ip = self.get_container_ip(container_id)
        container_port = 8000
        container_url = f"http://{container_ip}:{container_port}"
        
        try:
            response = requests.get(
                f"{container_url}/get_tool_list",
                timeout=1,
                headers={'Accept': 'application/json'}
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError:
            raise Exception(f"Container service not available for {container_id}")
        except requests.exceptions.Timeout:
            raise Exception(f"Request timeout getting tool list for {container_id}")
        except Exception as e:
            raise Exception(f"Failed to get tool list: {str(e)}")

    def call_container_tool(self, container_id: str, tool_name: str, tool_args: Dict[str, Any]) -> Dict[str, Any]:
        import requests

        container_ip = self.get_container_ip(container_id)
        container_port = 8000
        container_url = f"http://{container_ip}:{container_port}"

        request_data = {"tool": tool_name, "args": tool_args}

        try:
            response = requests.post(
                f"{container_url}/run",
                json=request_data,
                timeout=30,
                headers={
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                }
            )

            if response.status_code == 200:
                return response.json()
            else:
                try:
                    error_response = response.json()
                    error_detail = error_response.get('detail', f'HTTP {response.status_code} error')
                except:
                    error_detail = f'HTTP {response.status_code} error'

                return {
                    "status": "error",
                    "error": error_detail,
                    "http_status": response.status_code
                }

        except requests.exceptions.ConnectionError:
            return {
                "status": "error",
                "error": f"container service not available for {container_id}"
            }
        except requests.exceptions.Timeout:
            return {
                "status": "error",
                "error": f"request timeout calling tool {tool_name} in container {container_id}"
            }
        except Exception as e:
            return {
                "status": "error",
                "error": f"failed to call tool {tool_name}: {str(e)}"
            }

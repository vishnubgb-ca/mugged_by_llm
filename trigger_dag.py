import paramiko
import os
import requests
#airflow_instance_private_key = os.environ["airflow_instance_private_key"]
#airflow_instance_ip = os.environ["airflow_instance_ip"]
#print("airflow_instance_ip",airflow_instance_ip)
#ssh_client=paramiko.SSHClient()
#ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
#key = paramiko.RSAKey.from_private_key_file(airflow_instance_private_key)
#print("connecting to instance........")
#ssh_client.connect(hostname = airflow_instance_ip, username='ubuntu', pkey = key)
#print("connected")
#ftp_client=ssh_client.open_sftp()
#commands = ["sudo docker exec 7689635ac539 airflow dags trigger Supply-Chain-Management"]

# for cmd in commands:
#     stdin, stdout, stderr = ssh_client.exec_command(cmd)
#     print(stdout.read())
#     print(stderr.read())
deployment_url = os.environ["airflow_url"]
dag_id = mugged_by_llm
response = requests.post(
    url=f"{deployment_url}/api/v1/dags/{dag_id}/dagRuns",
    auth=(os.environ['airflow_username'], os.environ['airflow_password']),
    verify=False,
    json={
        "conf": {},
    }
)
print(response)
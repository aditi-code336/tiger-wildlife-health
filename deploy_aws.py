import os
import sys
import json
import time
import subprocess
import shutil

def run_cmd(cmd, check=True):
    print(f"[EXEC] {cmd}")
    res = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    if res.stdout:
        print(res.stdout.strip())
    if res.stderr:
        print(f"[STDERR] {res.stderr.strip()}", file=sys.stderr)
    if check and res.returncode != 0:
        raise RuntimeError(f"Command failed with code {res.returncode}: {res.stderr.strip()}")
    return res

def check_aws_credentials():
    print("Verifying AWS CLI credentials...")
    res = subprocess.run("aws sts get-caller-identity", shell=True, text=True, capture_output=True)
    if res.returncode == 0:
        try:
            data = json.loads(res.stdout)
            print(f"[AUTH SUCCESS] Connected to AWS Account: {data.get('Account')} | Arn: {data.get('Arn')}")
            return True, data
        except Exception:
            return True, None
    else:
        err = res.stderr.strip()
        print(f"[AUTH FAILED] AWS STS check failed:\n{err}")
        return False, err

def deploy_to_ec2(region="us-east-1"):
    print(f"\n--- Initiating Automated AWS EC2 Deployment in {region} ---")
    
    # 1. Get default VPC
    res = run_cmd(f"aws ec2 describe-vpcs --filters Name=is-default,Values=true --region {region}")
    vpcs = json.loads(res.stdout).get("Vpcs", [])
    if not vpcs:
        res = run_cmd(f"aws ec2 describe-vpcs --region {region}")
        vpcs = json.loads(res.stdout).get("Vpcs", [])
    if not vpcs:
        raise RuntimeError("No VPC found in your AWS account for this region.")
    vpc_id = vpcs[0]["VpcId"]
    print(f"Using VPC: {vpc_id}")

    # 2. Create Security Group
    sg_name = f"tiger-health-sg-{int(time.time())}"
    print(f"Creating Security Group {sg_name}...")
    res = run_cmd(f'aws ec2 create-security-group --group-name {sg_name} --description "Tiger Health System SG" --vpc-id {vpc_id} --region {region}')
    sg_id = json.loads(res.stdout)["GroupId"]
    print(f"Created Security Group: {sg_id}")

    # Authorize port 8001 (App), port 80 (HTTP), and port 22 (SSH)
    print("Opening ports 8001, 80, and 22...")
    run_cmd(f'aws ec2 authorize-security-group-ingress --group-id {sg_id} --protocol tcp --port 8001 --cidr 0.0.0.0/0 --region {region}')
    run_cmd(f'aws ec2 authorize-security-group-ingress --group-id {sg_id} --protocol tcp --port 80 --cidr 0.0.0.0/0 --region {region}', check=False)
    run_cmd(f'aws ec2 authorize-security-group-ingress --group-id {sg_id} --protocol tcp --port 22 --cidr 0.0.0.0/0 --region {region}', check=False)

    # 3. Read user data script
    user_data_path = os.path.join(os.path.dirname(__file__), "aws_ec2_setup.sh")
    with open(user_data_path, "r", encoding="utf-8") as f:
        user_data_content = f.read()

    # 4. Resolve Ubuntu 24.04 or 22.04 AMI
    print("Locating latest Ubuntu AMI...")
    res = run_cmd(f'aws ec2 describe-images --owners 099720109477 --filters "Name=name,Values=ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*" "Name=state,Values=available" --query "reverse(sort_by(Images, &CreationDate))[0].ImageId" --output text --region {region}')
    ami_id = res.stdout.strip()
    print(f"Selected AMI: {ami_id}")

    # 5. Launch EC2 Instance (t2.micro / t3.micro Free Tier)
    instance_type = "t2.micro" if region in ["us-east-1", "us-east-2", "us-west-2"] else "t3.micro"
    print(f"Launching {instance_type} EC2 instance...")

    # Write user data to temp file for CLI
    tmp_ud = "temp_user_data.sh"
    with open(tmp_ud, "w", encoding="utf-8") as f:
        f.write(user_data_content)

    launch_cmd = (
        f'aws ec2 run-instances --image-id {ami_id} --count 1 --instance-type {instance_type} '
        f'--security-group-ids {sg_id} --user-data file://{tmp_ud} '
        f'--tag-specifications "ResourceType=instance,Tags=[{{Key=Name,Value=Tiger-Wildlife-Health-Diagnostics}}]" '
        f'--region {region}'
    )
    res = run_cmd(launch_cmd)
    if os.path.exists(tmp_ud):
        os.remove(tmp_ud)

    instance_id = json.loads(res.stdout)["Instances"][0]["InstanceId"]
    print(f"[LAUNCHED] Instance ID: {instance_id}")

    print("Waiting for instance public IP...")
    time.sleep(10)
    for _ in range(12):
        desc = run_cmd(f'aws ec2 describe-instances --instance-ids {instance_id} --region {region}')
        inst_info = json.loads(desc.stdout)["Reservations"][0]["Instances"][0]
        public_ip = inst_info.get("PublicIpAddress")
        state = inst_info.get("State", {}).get("Name")
        if public_ip:
            print(f"\n=======================================================")
            print(f"AWS EC2 INSTANCE LIVE!")
            print(f"Instance ID: {instance_id}")
            print(f"Public IP:   {public_ip}")
            print(f"Live URL:    http://{public_ip}:8001/")
            print(f"=======================================================\n")
            print("Note: The instance is currently building the Docker container from GitHub.")
            print(f"The web interface will be ready in ~90 seconds at: http://{public_ip}:8001/")
            return public_ip
        print(f"State: {state} ... waiting for public IP allocation...")
        time.sleep(5)

    return instance_id

def main():
    print("="*60)
    print("TIGER WILDLIFE HEALTH DIAGNOSTIC SYSTEM - AWS DEPLOYER")
    print("="*60)

    is_authed, auth_info = check_aws_credentials()
    if not is_authed:
        print("\n" + "!"*60)
        print("ACTION REQUIRED: AWS Credentials Not Valid")
        print("!"*60)
        print("\nReason:")
        print("Your AWS credentials in ~/.aws/credentials were rejected by AWS with:")
        print(f"  {auth_info}\n")
        print("Common Solutions:")
        print("1. If using AWS Academy / Learner Lab:")
        print("   In your Learner Lab, click 'AWS Details', then click 'Show' next to AWS CLI credentials.")
        print("   Copy the ENTIRE block (it includes aws_access_key_id, aws_secret_access_key, AND aws_session_token).")
        print("   Open ~/.aws/credentials and paste the entire block.\n")
        print("2. If using standard AWS IAM Account:")
        print("   Run: aws configure")
        print("   Re-enter your AWS Access Key ID and Secret Access Key carefully without extra spaces.\n")
        print("3. Alternatively: Use AWS App Runner via Console (Zero CLI Setup needed!):")
        print("   Go to https://console.aws.amazon.com/apprunner")
        print("   Click 'Create Service' -> Source: GitHub -> select 'aditi-code336/tiger-wildlife-health'")
        print("   Select 'Use Dockerfile', set port 8001, and click Deploy!")
        print("="*60)
        sys.exit(1)

    region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
    deploy_to_ec2(region=region)

if __name__ == '__main__':
    main()

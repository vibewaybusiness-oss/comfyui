The file start-service.sh must be moved to the workspace and 
RunPod lets you specify a Container Command (startup command) in the template.
In your Pod Template settings → set startup command to:

bash /workspace/start-services.sh


___________________________________________________________________________________________________________________________________________________________
RUNPOD INSTALL

Create account
-> Storage Create
-> Create Secrets for Network Volume

###-verbose
ssh -v soo39scsjo4q15-64410ea6@ssh.runpod.io -i ~/.ssh/id_ed25519 

###Copy Paste
scp -i ~/.ssh/id_ed25519 code/vibewave/api/runpod/workspace_setup/init.sh 4aze3v4osuggky-64411002@ssh.runpod.io:/workspace/
scp -i ~/.ssh/id_ed25519 code/vibewave/api/runpod/workspace_setup/start-service.sh 4aze3v4osuggky-64411002@ssh.runpod.io:/workspace/

scp -i ~/.ssh/id_ed25519 code/comfyui-runpod/comfyui/models/ 4aze3v4osuggky-64411002@ssh.runpod.io:/workspace/comfyui/models/
scp -i ~/.ssh/id_ed25519 code/comfyui-runpod/comfyui/user/ 4aze3v4osuggky-64411002@ssh.runpod.io:/workspace/comfyui/user/


SetupComfyUI from git - Runpod SSH access:

ssh-keygen -t ed25519 -C "vibeway.business@gmail.com"
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519

(show key)
cat ~/.ssh/id_ed25519.pub

____
--> Wait for https://github.com/settings/ssh
Setup the SSH key in the git profile
Verify email
____

(check access)
ssh -T git@github.com

git clone git@github.com:vibewaybusiness-oss/comfyui.git


-------------------------------------------------

Container image:
docker.io/library/nginx:latest

Container start command:
cd /workspace && bash  start-service.sh

-------------------------------------------------

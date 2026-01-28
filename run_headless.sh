xhost +SI:localuser:root
docker run --rm -it --gpus all --net=host --name="rlbench" \
  -e DISPLAY=:99 \
  -e NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v /home/yigit/projects/inverse/reversible_skills/scripts:/workspace/scripts:rw \
  -v /home/yigit/projects/inverse/reversible_skills/data:/workspace/data:rw \
  rlbench:20.04-gpu

"""Enter a private, user-owned mount namespace with the validated runtime."""
import os
import sys

base = '/home/lwx/.cache/megatron-gdn-20261026'
venv = '/home/songbinbin/.venvs/megatron-gdn-offload'
source = os.environ.get('GDN_SOURCE', base + '/source')
command = ['bwrap', '--die-with-parent', '--ro-bind', base + '/rootfs', '/',
           '--bind', base, base, '--ro-bind', base + '/venv', venv,
           '--ro-bind', base + '/python', '/data/songbinbin/minicoda',
           '--proc', '/proc', '--ro-bind', '/sys', '/sys', '--dev', '/dev',
           '--tmpfs', '/tmp', '--ro-bind', '/etc/resolv.conf', '/etc/resolv.conf']
for device in ('nvidia1', 'nvidiactl', 'nvidia-uvm', 'nvidia-uvm-tools'):
    command += ['--dev-bind', '/dev/' + device, '/dev/' + device]
for library in ('libcuda.so.1', 'libnvidia-ml.so.1', 'libnvidia-nvvm.so.4', 'libnvidia-ptxjitcompiler.so.1'):
    command += ['--ro-bind', '/lib/x86_64-linux-gnu/' + library, '/opt/nvidia/' + library]
for name, value in dict(
    CUDA_VISIBLE_DEVICES='GPU-5ceca9be-9cf0-99be-0868-915db7531fae',
    LD_LIBRARY_PATH='/opt/nvidia', PYTHONDONTWRITEBYTECODE='1', HOME=base, XDG_CACHE_HOME=base+'/cache',
    PYTHONPATH=':'.join([source, base, base+'/imports/flash-attn-python', base+'/imports/causal-conv1d-python', base+'/imports/bridge-python', base+'/imports/src', base+'/apex-wgrad-build/out']),
    LIFECYCLE_OUTPUT=base+'/precision-observations',
    NVTE_NVFP4_DISABLE_RHT='0', NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING='0',
    NVTE_FLASH_ATTN=os.environ.get('NVTE_FLASH_ATTN', '1'),
    CUDA_DEVICE_MAX_CONNECTIONS='8', NCCL_NVLS_ENABLE='0', NCCL_MAX_NCHANNELS='1',
    NVTE_CPU_OFFLOAD_V1='1', OMP_NUM_THREADS='1',
    TRITON_CACHE_DIR=os.environ['GDN_TRITON_CACHE_DIR'],
    TORCHINDUCTOR_CACHE_DIR=os.environ['GDN_INDUCTOR_CACHE_DIR'],
    PATH=base + '/apex-wgrad-build/cuda/nvvm/bin:' + base + '/apex-wgrad-build/cuda/bin:' + venv + '/bin:/usr/local/bin:/usr/bin:/bin', VIRTUAL_ENV=venv,
).items():
    command += ['--setenv', name, value]
command += ['--chdir', source, '--', *sys.argv[1:]]
os.execvp(command[0], command)

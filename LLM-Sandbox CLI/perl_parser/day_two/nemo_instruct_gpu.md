Based on the provided Perl module examples, here's how you might integrate them to manage GPU-intensive tasks like running llama.cpp or Ollama models while monitoring GPU usage on Ubuntu 22.04:

1. **Create a script for managing processes:**

```perl
#!/usr/bin/perl

use strict;
use warnings;

use IPC::Open2;
use Time::HiRes qw(sleep);

my %processes = (
    llama => {
        cmd => 'llama.cpp',
        args => ['--model_path', '/path/to/model', '--input', 'example.txt'],
    },
    ollama => {
        cmd => 'ollama',
        args => [],
    },
);

sub run_process {
    my ($name, $pid) = @_;
    my ($READER, $WRITER);
    open2($READER, $WRITER, $processes{$name}{cmd}, @{ $processes{$name}{args} }) or die "Cannot fork: $!";

    print "Running $name process with PID: $pid\n";
    while (<$READER>) {
        print "$name output: $_";
    }
    close($WRITER);
}

sub monitor_gpu_usage {
    my $gpu_info = `nvidia-smi`;
    print "GPU Info:\n$gpu_info";
}

my %pids;

for my $process (sort keys %processes) {
    if ($pids{$process}) {
        waitpid($pids{$process}, 0);
    }
    $pids{$process} = fork();
    if (!$pids{$process}) {
        run_process($process, $$);
        exit(0);
    }

    sleep(1); # Allow some time for the process to start
}

while (1) {
    monitor_gpu_usage();
    sleep(5); # Check GPU usage every 5 seconds

    for my $process (sort keys %pids) {
        if (!waitpid($pids{$process}, WNOHANG)) {
            print "Process $process is still running\n";
        } else {
            delete $pids{$process};
            print "$process process exited\n";
        }
    }

    last if !keys %pids;
}
```

2. **Make the script executable:**

```bash
chmod +x manage_gpu_processes.pl
```

3. **Run the script:**

```bash
./manage_gpu_processes.pl
```

This script will run both llama.cpp and Ollama processes simultaneously, monitor GPU usage periodically using `nvidia-smi`, and wait for both processes to complete before exiting. You can adjust the `$processes` hash based on your specific requirements.

To integrate other modules like `Parallel::ForkManager` or change the process management strategy, you can modify the script accordingly following the examples provided earlier.


Sure! Here are NVIDIA Container Toolkit (nvidia-ctk) examples for running Ollama and llama.cpp on NVIDIA GPUs using Docker. Make sure you have Docker installed and configured with GPU support.

### 1. Ollama

Create a `Dockerfile`:

```Dockerfile
FROM nvidia/cuda:12.0-base-runtime-ubuntu2204

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get install -y wget git ca-certificates && \
    wget https://ollama.ai/downloads/ollama-1.13.6-linux-x86_64.tar.gz && \
    tar xvf ollama-*.tar.gz && \
    mv ollama /usr/local/bin/

WORKDIR /app
COPY . .

CMD ["ollama", "run", "/path/to/model", "--port", "7860"]
```

Build and run the Docker container:

```bash
docker build -t my ollama .
docker run --gpu 1 --ipc host -p 7860:7860 my-ollama
```

### 2. llama.cpp

Create a `Dockerfile`:

```Dockerfile
FROM nvidia/cuda:12.0-base-runtime-ubuntu2204

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get install -y git cmake ninja-build libopenmp-dev && \
    git clone https://github.com/ggerganov/llama.cpp.git && \
    cd llama.cpp && \
    mkdir build && \
    cd build && \
    cmake .. -B. -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=ON && \
    ninja && \
    cd ../..

WORKDIR /app
COPY . .

CMD ["./build/examples/main", "--model_path", "/path/to/model", "--input", "example.txt"]
```

Build and run the Docker container:

```bash
docker build -t my-llama-cpp .
docker run --gpu 1 --ipc host my-llama-cpp
```

Replace `/path/to/model` with the actual path to your model in both examples.

These Dockerfiles use NVIDIA's official CUDA base image and install the required dependencies for running Ollama or building llama.cpp. The resulting containers will have GPU support enabled, allowing you to run these models on NVIDIA GPUs.<|im_end|>

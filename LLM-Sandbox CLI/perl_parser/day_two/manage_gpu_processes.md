Here is a more detailed Perl script based on your use case for managing GPU-intensive tasks like running `llama.cpp` or `Ollama` models while monitoring GPU usage on Ubuntu 22.04:

```perl
#!/usr/bin/perl

use strict;
use warnings;
use IPC::Open2;
use Time::HiRes qw(sleep);

my %processes = (
    llama => {
        cmd  => 'llama.cpp',
        args => ['--model_path', '/path/to/model', '--input', 'example.txt'],
    },
    ollama => {
        cmd  => 'ollama',
        args => [],
    },
);

sub run_process {
    my ($name, $pid) = @_;
    my ($READER, $WRITER);
    my $cmd = join(' ', $processes{$name}{cmd}, @{ $processes{$name}{args} });
    my $child_pid = open2($READER, $WRITER, $cmd) or die "Cannot fork: $!";

    print "Running $name process with PID: $child_pid\n";
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
        if (waitpid($pids{$process}, WNOHANG) == 0) {
            print "Process $process is still running\n";
        } else {
            delete $pids{$process};
            print "$process process exited\n";
        }
    }

    last if !keys %pids;
}
```

### Steps to Run the Script

1. **Make the script executable:**
   ```bash
   chmod +x manage_gpu_processes.pl
   ```

2. **Run the script:**
   ```bash
   ./manage_gpu_processes.pl
   ```

### Explanation

- **Process Management:**
  - The `%processes` hash defines the commands and arguments for `llama.cpp` and `Ollama`.
  - The `run_process` subroutine uses `IPC::Open2` to fork a child process for each command and manage input/output.
  - The `monitor_gpu_usage` subroutine retrieves and prints the current GPU usage using `nvidia-smi`.

- **Main Loop:**
  - Forks each process defined in the `%processes` hash and runs them.
  - Monitors GPU usage every 5 seconds using the `monitor_gpu_usage` subroutine.
  - Checks if any processes have exited and removes their PID from the `%pids` hash.

This script provides a basic framework for managing and monitoring GPU-intensive tasks in Perl. You can extend it to handle more complex scenarios or integrate additional modules like `Parallel::ForkManager` for more advanced process management.

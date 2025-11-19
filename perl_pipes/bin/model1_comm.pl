#!/usr/bin/env perl
use v5.36;
use warnings;
use autodie;
use FindBin qw($RealBin);
use lib "$RealBin/../lib";
use Getopt::Long;
use Time::HiRes qw(time);

# Model 1 Communication Script with Enhanced Features
# Inter-Process Communication for Multi-Model LLM Sessions

our $VERSION = '2.0.0';

# Configuration
my %config = (
    pipe_to_model2   => $ENV{PIPE_TO_MODEL2}   // '/tmp/model1_to_model2',
    pipe_from_model2 => $ENV{PIPE_FROM_MODEL2} // '/tmp/model2_to_model1',
    timeout          => $ENV{PIPE_TIMEOUT}     // 30,
    verbose          => 0,
    log_file         => $ENV{LOG_FILE}         // "$RealBin/../logs/model1.log",
);

# Parse command line options
GetOptions(
    'to=s'       => \$config{pipe_to_model2},
    'from=s'     => \$config{pipe_from_model2},
    'timeout=i'  => \$config{timeout},
    'verbose'    => \$config{verbose},
    'log=s'      => \$config{log_file},
    'help'       => \my $help,
    'version'    => \my $version,
) or die "Error in command line arguments\n";

if ($help) {
    print_help();
    exit 0;
}

if ($version) {
    say "Model 1 Communication Script v$VERSION";
    exit 0;
}

# Initialize logging
mkdir "$RealBin/../logs" unless -d "$RealBin/../logs";
open my $log_fh, '>>', $config{log_file};
$log_fh->autoflush(1);

sub log_message($level, $message) {
    my $timestamp = scalar localtime;
    my $log_line = "[$timestamp] [$level] $message\n";
    print $log_fh $log_line;
    say STDERR $log_line if $config{verbose};
}

sub print_help() {
    say <<'HELP';
Model 1 Communication Script - IPC for Multi-Model LLM Sessions

Usage: model1_comm.pl [OPTIONS]

Options:
    --to=PATH       Path to pipe for sending to Model 2 (default: /tmp/model1_to_model2)
    --from=PATH     Path to pipe for receiving from Model 2 (default: /tmp/model2_to_model1)
    --timeout=SEC   Timeout in seconds (default: 30)
    --verbose       Enable verbose output
    --log=FILE      Log file path
    --help          Show this help message
    --version       Show version information

Environment Variables:
    PIPE_TO_MODEL2      Override default output pipe
    PIPE_FROM_MODEL2    Override default input pipe
    PIPE_TIMEOUT        Override default timeout
    LOG_FILE            Override default log file

Examples:
    # Basic usage
    ./model1_comm.pl

    # Custom pipes with verbose output
    ./model1_comm.pl --to=/tmp/custom_out --from=/tmp/custom_in --verbose

    # Read message from stdin
    echo "Hello Model 2" | ./model1_comm.pl
HELP
}

sub send_message($pipe, $message) {
    log_message('INFO', "Opening pipe for writing: $pipe");

    # Use alarm for timeout
    eval {
        local $SIG{ALRM} = sub { die "Timeout\n" };
        alarm $config{timeout};

        open my $out, '>', $pipe;
        $out->autoflush(1);
        print $out $message;
        close $out;

        alarm 0;
    };

    if ($@) {
        if ($@ =~ /Timeout/) {
            log_message('ERROR', "Timeout writing to pipe: $pipe");
            die "Timeout sending message\n";
        }
        die $@;
    }

    log_message('INFO', "Message sent successfully");
}

sub receive_message($pipe) {
    log_message('INFO', "Opening pipe for reading: $pipe");

    my $response;
    eval {
        local $SIG{ALRM} = sub { die "Timeout\n" };
        alarm $config{timeout};

        open my $in, '<', $pipe;
        $response = <$in>;
        close $in;

        alarm 0;
    };

    if ($@) {
        if ($@ =~ /Timeout/) {
            log_message('ERROR', "Timeout reading from pipe: $pipe");
            die "Timeout receiving message\n";
        }
        die $@;
    }

    log_message('INFO', "Message received successfully");
    return $response;
}

# Main execution
sub main() {
    log_message('INFO', "Starting Model 1 Communication (v$VERSION)");
    log_message('INFO', "Output pipe: $config{pipe_to_model2}");
    log_message('INFO', "Input pipe: $config{pipe_from_model2}");

    # Verify pipes exist
    unless (-p $config{pipe_to_model2}) {
        log_message('ERROR', "Output pipe does not exist: $config{pipe_to_model2}");
        die "Pipe not found: $config{pipe_to_model2}\nRun setup_pipes.sh first\n";
    }

    unless (-p $config{pipe_from_model2}) {
        log_message('ERROR', "Input pipe does not exist: $config{pipe_from_model2}");
        die "Pipe not found: $config{pipe_from_model2}\nRun setup_pipes.sh first\n";
    }

    # Read message from stdin or use default
    my $message;
    if (-t STDIN) {
        $message = "Hello from Model 1 at " . scalar(localtime) . "\n";
    } else {
        $message = do { local $/; <STDIN> };
    }

    log_message('INFO', "Sending message: $message");

    my $start_time = time;

    # Send message to Model 2
    send_message($config{pipe_to_model2}, $message);

    # Receive response from Model 2
    my $response = receive_message($config{pipe_from_model2});

    my $elapsed = time - $start_time;

    log_message('INFO', "Received response: $response");
    log_message('INFO', sprintf("Round-trip time: %.3f seconds", $elapsed));

    # Output response
    print "Model 1 received: $response";
    say STDERR sprintf("Round-trip time: %.3f seconds", $elapsed) if $config{verbose};

    log_message('INFO', "Communication completed successfully");
}

# Run main program
eval {
    main();
};

if ($@) {
    log_message('ERROR', "Fatal error: $@");
    die $@;
}

close $log_fh;

__END__

=head1 NAME

model1_comm.pl - Model 1 IPC Communication Script

=head1 SYNOPSIS

    model1_comm.pl [OPTIONS]

    echo "message" | model1_comm.pl --verbose

=head1 DESCRIPTION

This script enables Model 1 to communicate with Model 2 via named pipes (FIFOs)
for multi-model LLM session orchestration. It supports timeout handling, logging,
and configurable pipe paths.

=head1 AUTHOR

GRYPHGEN Project - Multi-Model LLM Communication

=head1 LICENSE

MIT License

=cut

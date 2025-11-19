#!/usr/bin/env perl
use v5.36;
use warnings;
use autodie;
use FindBin qw($RealBin);
use lib "$RealBin/../lib";
use Getopt::Long;
use Time::HiRes qw(time);

# Model 2 Communication Script with Enhanced Features
# Inter-Process Communication for Multi-Model LLM Sessions

our $VERSION = '2.0.0';

# Configuration
my %config = (
    pipe_to_model1   => $ENV{PIPE_TO_MODEL1}   // '/tmp/model2_to_model1',
    pipe_from_model1 => $ENV{PIPE_FROM_MODEL1} // '/tmp/model1_to_model2',
    timeout          => $ENV{PIPE_TIMEOUT}     // 30,
    verbose          => 0,
    log_file         => $ENV{LOG_FILE}         // "$RealBin/../logs/model2.log",
);

# Parse command line options
GetOptions(
    'to=s'       => \$config{pipe_to_model1},
    'from=s'     => \$config{pipe_from_model1},
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
    say "Model 2 Communication Script v$VERSION";
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
Model 2 Communication Script - IPC for Multi-Model LLM Sessions

Usage: model2_comm.pl [OPTIONS]

Options:
    --to=PATH       Path to pipe for sending to Model 1 (default: /tmp/model2_to_model1)
    --from=PATH     Path to pipe for receiving from Model 1 (default: /tmp/model1_to_model2)
    --timeout=SEC   Timeout in seconds (default: 30)
    --verbose       Enable verbose output
    --log=FILE      Log file path
    --help          Show this help message
    --version       Show version information

Environment Variables:
    PIPE_TO_MODEL1      Override default output pipe
    PIPE_FROM_MODEL1    Override default input pipe
    PIPE_TIMEOUT        Override default timeout
    LOG_FILE            Override default log file

Examples:
    # Basic usage
    ./model2_comm.pl

    # Custom pipes with verbose output
    ./model2_comm.pl --to=/tmp/custom_out --from=/tmp/custom_in --verbose
HELP
}

sub receive_message($pipe) {
    log_message('INFO', "Opening pipe for reading: $pipe");

    my $message;
    eval {
        local $SIG{ALRM} = sub { die "Timeout\n" };
        alarm $config{timeout};

        open my $in, '<', $pipe;
        $message = <$in>;
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
    return $message;
}

sub send_message($pipe, $message) {
    log_message('INFO', "Opening pipe for writing: $pipe");

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

# Main execution
sub main() {
    log_message('INFO', "Starting Model 2 Communication (v$VERSION)");
    log_message('INFO', "Input pipe: $config{pipe_from_model1}");
    log_message('INFO', "Output pipe: $config{pipe_to_model1}");

    # Verify pipes exist
    unless (-p $config{pipe_from_model1}) {
        log_message('ERROR', "Input pipe does not exist: $config{pipe_from_model1}");
        die "Pipe not found: $config{pipe_from_model1}\nRun setup_pipes.sh first\n";
    }

    unless (-p $config{pipe_to_model1}) {
        log_message('ERROR', "Output pipe does not exist: $config{pipe_to_model1}");
        die "Pipe not found: $config{pipe_to_model1}\nRun setup_pipes.sh first\n";
    }

    my $start_time = time;

    # Receive message from Model 1
    my $message = receive_message($config{pipe_from_model1});

    log_message('INFO', "Received message: $message");
    print "Model 2 received: $message";

    # Generate response
    my $response = "Hello from Model 2 (responding at " . scalar(localtime) . ")\n";

    log_message('INFO', "Sending response: $response");

    # Send response to Model 1
    send_message($config{pipe_to_model1}, $response);

    my $elapsed = time - $start_time;

    log_message('INFO', sprintf("Processing time: %.3f seconds", $elapsed));
    say STDERR sprintf("Processing time: %.3f seconds", $elapsed) if $config{verbose};

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

model2_comm.pl - Model 2 IPC Communication Script

=head1 SYNOPSIS

    model2_comm.pl [OPTIONS]

=head1 DESCRIPTION

This script enables Model 2 to receive messages from Model 1 and send responses
via named pipes (FIFOs) for multi-model LLM session orchestration.

=head1 AUTHOR

GRYPHGEN Project - Multi-Model LLM Communication

=head1 LICENSE

MIT License

=cut

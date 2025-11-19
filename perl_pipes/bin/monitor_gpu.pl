#!/usr/bin/env perl
use v5.36;
use warnings;
use Getopt::Long;
use Time::HiRes qw(sleep time);
use POSIX qw(strftime);

# GPU Monitoring Script for Multi-Model LLM Sessions
# Optimized for NVIDIA RTX 4080 16GB

our $VERSION = '2.0.0';

# Configuration
my %config = (
    interval     => 5,
    format       => 'table',  # table, csv, json
    output_file  => undef,
    alert_vram   => 95,       # Alert if VRAM usage > 95%
    alert_temp   => 85,       # Alert if temp > 85°C
    verbose      => 0,
    continuous   => 0,
);

GetOptions(
    'interval=i'   => \$config{interval},
    'format=s'     => \$config{format},
    'output=s'     => \$config{output_file},
    'alert-vram=i' => \$config{alert_vram},
    'alert-temp=i' => \$config{alert_temp},
    'verbose'      => \$config{verbose},
    'continuous'   => \$config{continuous},
    'help'         => \my $help,
    'version'      => \my $version,
) or die "Error in command line arguments\n";

if ($help) {
    print_help();
    exit 0;
}

if ($version) {
    say "GPU Monitor v$VERSION";
    exit 0;
}

sub print_help() {
    say <<'HELP';
GPU Monitoring Script - Real-time NVIDIA GPU Statistics

Usage: monitor_gpu.pl [OPTIONS]

Options:
    --interval=SEC      Monitoring interval in seconds (default: 5)
    --format=FORMAT     Output format: table, csv, json (default: table)
    --output=FILE       Write output to file
    --alert-vram=PCT    Alert threshold for VRAM usage % (default: 95)
    --alert-temp=C      Alert threshold for temperature (default: 85)
    --continuous        Run continuously (Ctrl+C to stop)
    --verbose           Enable verbose output
    --help              Show this help message
    --version           Show version information

Examples:
    # Monitor once
    ./monitor_gpu.pl

    # Continuous monitoring every 2 seconds
    ./monitor_gpu.pl --interval=2 --continuous

    # Log to file in CSV format
    ./monitor_gpu.pl --format=csv --output=gpu_stats.csv --continuous

    # Alert on high VRAM usage
    ./monitor_gpu.pl --alert-vram=90 --continuous

Requires:
    - NVIDIA GPU
    - nvidia-smi command
HELP
}

sub check_nvidia_smi() {
    my $result = `which nvidia-smi 2>/dev/null`;
    chomp $result;
    return $result ne '';
}

sub get_gpu_stats() {
    my $query = 'timestamp,name,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw,clocks.current.graphics';
    my $output = `nvidia-smi --query-gpu=$query --format=csv,noheader,nounits 2>/dev/null`;

    return undef unless $output;

    chomp $output;
    my @fields = split /,\s*/, $output;

    return {
        timestamp   => $fields[0] // '',
        name        => $fields[1] // 'Unknown',
        temp        => $fields[2] // 0,
        util        => $fields[3] // 0,
        mem_used    => $fields[4] // 0,
        mem_total   => $fields[5] // 0,
        power       => $fields[6] // 0,
        clock       => $fields[7] // 0,
    };
}

sub calculate_derived_stats($stats) {
    $stats->{mem_free} = $stats->{mem_total} - $stats->{mem_used};
    $stats->{mem_used_pct} = $stats->{mem_total} > 0
        ? sprintf("%.1f", ($stats->{mem_used} / $stats->{mem_total}) * 100)
        : 0;
    $stats->{mem_free_pct} = $stats->{mem_total} > 0
        ? sprintf("%.1f", ($stats->{mem_free} / $stats->{mem_total}) * 100)
        : 0;

    return $stats;
}

sub check_alerts($stats) {
    my @alerts;

    if ($stats->{mem_used_pct} > $config{alert_vram}) {
        push @alerts, sprintf("HIGH VRAM: %.1f%% (threshold: %d%%)",
            $stats->{mem_used_pct}, $config{alert_vram});
    }

    if ($stats->{temp} > $config{alert_temp}) {
        push @alerts, sprintf("HIGH TEMP: %d°C (threshold: %d°C)",
            $stats->{temp}, $config{alert_temp});
    }

    return @alerts;
}

sub format_table($stats) {
    my $output = "\n" . "=" x 80 . "\n";
    $output .= "GPU Monitor - " . strftime("%Y-%m-%d %H:%M:%S", localtime) . "\n";
    $output .= "=" x 80 . "\n\n";

    $output .= sprintf("GPU:          %s\n", $stats->{name});
    $output .= sprintf("Temperature:  %d°C\n", $stats->{temp});
    $output .= sprintf("Utilization:  %d%%\n", $stats->{util});
    $output .= sprintf("Clock Speed:  %d MHz\n", $stats->{clock});
    $output .= sprintf("Power Draw:   %.1f W\n\n", $stats->{power});

    $output .= sprintf("VRAM Usage:   %d / %d MB (%.1f%%)\n",
        $stats->{mem_used}, $stats->{mem_total}, $stats->{mem_used_pct});
    $output .= sprintf("VRAM Free:    %d MB (%.1f%%)\n",
        $stats->{mem_free}, $stats->{mem_free_pct});

    # Visual memory bar
    my $bar_width = 50;
    my $filled = int($stats->{mem_used_pct} / 100 * $bar_width);
    my $empty = $bar_width - $filled;
    $output .= "\n[" . ("█" x $filled) . ("░" x $empty) . "]\n";

    # Alerts
    my @alerts = check_alerts($stats);
    if (@alerts) {
        $output .= "\n⚠️  ALERTS:\n";
        for my $alert (@alerts) {
            $output .= "  - $alert\n";
        }
    }

    $output .= "\n" . "=" x 80 . "\n";

    return $output;
}

sub format_csv($stats) {
    state $header_printed = 0;

    my $output = '';

    unless ($header_printed) {
        $output .= "Timestamp,GPU,Temp(C),Util(%),Clock(MHz),Power(W),";
        $output .= "VRAM_Used(MB),VRAM_Total(MB),VRAM_Used(%),VRAM_Free(MB)\n";
        $header_printed = 1;
    }

    $output .= sprintf("%s,%s,%d,%d,%d,%.1f,%d,%d,%.1f,%d\n",
        strftime("%Y-%m-%d %H:%M:%S", localtime),
        $stats->{name},
        $stats->{temp},
        $stats->{util},
        $stats->{clock},
        $stats->{power},
        $stats->{mem_used},
        $stats->{mem_total},
        $stats->{mem_used_pct},
        $stats->{mem_free}
    );

    return $output;
}

sub format_json($stats) {
    $stats->{timestamp_iso} = strftime("%Y-%m-%dT%H:%M:%S", localtime);

    my $json = "{\n";
    $json .= qq{  "timestamp": "$stats->{timestamp_iso}",\n};
    $json .= qq{  "gpu": "$stats->{name}",\n};
    $json .= qq{  "temperature_c": $stats->{temp},\n};
    $json .= qq{  "utilization_pct": $stats->{util},\n};
    $json .= qq{  "clock_mhz": $stats->{clock},\n};
    $json .= qq{  "power_watts": $stats->{power},\n};
    $json .= qq{  "vram": {\n};
    $json .= qq{    "used_mb": $stats->{mem_used},\n};
    $json .= qq{    "total_mb": $stats->{mem_total},\n};
    $json .= qq{    "used_pct": $stats->{mem_used_pct},\n};
    $json .= qq{    "free_mb": $stats->{mem_free}\n};
    $json .= qq{  }\n};
    $json .= "}\n";

    return $json;
}

sub output_stats($stats, $output) {
    if ($config{output_file}) {
        open my $fh, '>>', $config{output_file}
            or die "Cannot open $config{output_file}: $!\n";
        print $fh $output;
        close $fh;

        say STDERR "Stats written to $config{output_file}" if $config{verbose};
    } else {
        print $output;
    }
}

# Main monitoring loop
sub main() {
    # Check for nvidia-smi
    unless (check_nvidia_smi()) {
        die "ERROR: nvidia-smi not found. Please install NVIDIA drivers.\n";
    }

    say STDERR "Starting GPU monitoring..." if $config{verbose};

    do {
        my $stats = get_gpu_stats();

        unless ($stats) {
            warn "Failed to get GPU statistics\n";
            sleep $config{interval};
            next;
        }

        $stats = calculate_derived_stats($stats);

        my $output;
        if ($config{format} eq 'csv') {
            $output = format_csv($stats);
        } elsif ($config{format} eq 'json') {
            $output = format_json($stats);
        } else {
            $output = format_table($stats);
        }

        output_stats($stats, $output);

        # Check alerts
        my @alerts = check_alerts($stats);
        if (@alerts && $config{verbose}) {
            for my $alert (@alerts) {
                warn "ALERT: $alert\n";
            }
        }

        sleep $config{interval} if $config{continuous};

    } while ($config{continuous});
}

# Run
main();

__END__

=head1 NAME

monitor_gpu.pl - NVIDIA GPU Monitoring for Multi-Model LLM Sessions

=head1 SYNOPSIS

    monitor_gpu.pl [OPTIONS]

    # Single check
    ./monitor_gpu.pl

    # Continuous monitoring
    ./monitor_gpu.pl --continuous --interval=2

=head1 DESCRIPTION

Real-time monitoring of NVIDIA GPU statistics optimized for tracking
multi-model LLM workloads on RTX 4080 16GB.

=head1 AUTHOR

GRYPHGEN Project

=head1 LICENSE

MIT License

=cut

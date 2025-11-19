#!/usr/bin/env perl
use v5.36;
use warnings;
use autodie;
use File::Find;
use File::stat;
use Getopt::Long;
use JSON::PP;
use POSIX qw(strftime);

# Enhanced GGUF Model File Finder with GPU Memory Awareness
# Optimized for NVIDIA RTX 4080 16GB

our $VERSION = '2.0.0';

# NVIDIA RTX 4080 specifications
my %GPU_SPECS = (
    name         => 'NVIDIA GeForce RTX 4080',
    vram_gb      => 16,
    vram_bytes   => 16 * 1024 * 1024 * 1024,
    cuda_cores   => 9728,
    tensor_cores => 304,
    architecture => 'Ada Lovelace',
);

# Configuration
my %config = (
    start_dir    => $ENV{SEARCH_PATH} // '/',
    format       => 'table',  # table, json, csv
    sort_by      => 'size',   # size, age, name
    show_gpu_fit => 1,
    min_size     => 0,
    max_size     => undef,
    max_age_days => undef,
    verbose      => 0,
);

# Parse command line options
GetOptions(
    'dir=s'        => \$config{start_dir},
    'format=s'     => \$config{format},
    'sort=s'       => \$config{sort_by},
    'gpu-fit!'     => \$config{show_gpu_fit},
    'min-size=s'   => \$config{min_size},
    'max-size=s'   => \$config{max_size},
    'max-age=i'    => \$config{max_age_days},
    'verbose'      => \$config{verbose},
    'help'         => \my $help,
    'version'      => \my $version,
) or die "Error in command line arguments\n";

if ($help) {
    print_help();
    exit 0;
}

if ($version) {
    say "GGUF File Finder v$VERSION";
    say "Optimized for $GPU_SPECS{name} ($GPU_SPECS{vram_gb}GB VRAM)";
    exit 0;
}

# Convert size strings like "1GB" to bytes
sub parse_size($size_str) {
    return $size_str if $size_str =~ /^\d+$/;

    if ($size_str =~ /^(\d+(?:\.\d+)?)\s*([KMGT]?B?)$/i) {
        my ($num, $unit) = ($1, uc($2));
        my %multipliers = (
            'B'  => 1,
            'KB' => 1024,
            'MB' => 1024 ** 2,
            'GB' => 1024 ** 3,
            'TB' => 1024 ** 4,
        );
        return int($num * ($multipliers{$unit} // 1));
    }
    die "Invalid size format: $size_str\n";
}

# Format file size in human-readable format
sub human_readable_size($size) {
    my @units = ('B', 'KB', 'MB', 'GB', 'TB');
    my $unit = 0;
    my $size_float = $size;

    while ($size_float > 1024 && $unit < @units - 1) {
        $size_float /= 1024;
        $unit++;
    }

    return sprintf("%.2f %s", $size_float, $units[$unit]);
}

# Check if model fits in GPU memory with quantization estimates
sub check_gpu_fit($size) {
    my %fit_status;

    # Direct fit check (with 90% usable VRAM assumption)
    my $usable_vram = $GPU_SPECS{vram_bytes} * 0.90;
    $fit_status{direct} = $size <= $usable_vram;

    # Estimate with context window overhead (typical 4K context ~500MB)
    my $with_context = $size + (500 * 1024 * 1024);
    $fit_status{with_context} = $with_context <= $usable_vram;

    # Calculate approximate model size in billions of parameters (rough estimate)
    # GGUF models: ~1GB per billion parameters for Q4 quantization
    my $estimated_params = $size / (1024 ** 3);
    $fit_status{estimated_params} = sprintf("~%.1fB", $estimated_params);

    return \%fit_status;
}

sub print_help() {
    say <<'HELP';
GGUF Model File Finder - GPU Memory Aware Search

Usage: find_gguf_files.pl [OPTIONS]

Options:
    --dir=PATH          Directory to search (default: /)
    --format=FORMAT     Output format: table, json, csv (default: table)
    --sort=FIELD        Sort by: size, age, name (default: size)
    --gpu-fit           Show GPU memory fit analysis (default: on)
    --no-gpu-fit        Disable GPU fit analysis
    --min-size=SIZE     Minimum file size (e.g., 1GB, 500MB)
    --max-size=SIZE     Maximum file size
    --max-age=DAYS      Maximum file age in days
    --verbose           Enable verbose output
    --help              Show this help message
    --version           Show version information

Environment Variables:
    SEARCH_PATH         Override default search directory

Examples:
    # Find all GGUF models
    ./find_gguf_files.pl

    # Search specific directory with size filter
    ./find_gguf_files.pl --dir=/mnt/models --max-size=15GB

    # Export to JSON
    ./find_gguf_files.pl --format=json > models.json

    # Find recent models that fit in GPU
    ./find_gguf_files.pl --max-age=30 --max-size=14GB

GPU Target:
    NVIDIA GeForce RTX 4080
    VRAM: 16GB
    Architecture: Ada Lovelace
HELP
}

# Array to store .gguf file paths and metadata
my @gguf_files;

# Subroutine to find .gguf files
sub find_gguf() {
    return unless -f $_ && $_ =~ /\.gguf$/i;

    my $file = $File::Find::name;
    my $stat = stat($file);
    my $size = $stat->size;
    my $mtime = $stat->mtime;
    my $age_in_days = int((time - $mtime) / (24 * 60 * 60));

    # Apply filters
    return if $config{min_size} && $size < parse_size($config{min_size});
    return if $config{max_size} && $size > parse_size($config{max_size});
    return if $config{max_age_days} && $age_in_days > $config{max_age_days};

    my $gpu_fit = $config{show_gpu_fit} ? check_gpu_fit($size) : undef;

    push @gguf_files, {
        path            => $file,
        size_bytes      => $size,
        size_human      => human_readable_size($size),
        age_days        => $age_in_days,
        mtime           => $mtime,
        mtime_str       => strftime("%Y-%m-%d %H:%M:%S", localtime($mtime)),
        gpu_fit         => $gpu_fit,
        basename        => (split('/', $file))[-1],
    };

    say STDERR "Found: $file (" . human_readable_size($size) . ")" if $config{verbose};
}

# Start the search
say STDERR "Searching for .gguf files in $config{start_dir}..." if $config{verbose};
find(\&find_gguf, $config{start_dir});

# Check if any files were found
unless (@gguf_files) {
    say "No .gguf files found matching criteria.";
    exit 0;
}

say STDERR "Found " . scalar(@gguf_files) . " files, sorting..." if $config{verbose};

# Sort files
my @sorted_files;
if ($config{sort_by} eq 'size') {
    @sorted_files = sort { $a->{size_bytes} <=> $b->{size_bytes} } @gguf_files;
} elsif ($config{sort_by} eq 'age') {
    @sorted_files = sort { $b->{age_days} <=> $a->{age_days} } @gguf_files;
} elsif ($config{sort_by} eq 'name') {
    @sorted_files = sort { $a->{basename} cmp $b->{basename} } @gguf_files;
} else {
    @sorted_files = @gguf_files;
}

# Output results
if ($config{format} eq 'json') {
    my $json = JSON::PP->new->pretty->encode({
        search_params => \%config,
        gpu_specs     => \%GPU_SPECS,
        total_files   => scalar(@sorted_files),
        files         => \@sorted_files,
    });
    print $json;
}
elsif ($config{format} eq 'csv') {
    say "Path,Size (bytes),Size (human),Age (days),Modified,GPU Fit,Est. Params";
    for my $file (@sorted_files) {
        my $gpu_status = $file->{gpu_fit}
            ? ($file->{gpu_fit}{with_context} ? 'Yes' : 'Tight')
            : 'N/A';
        my $params = $file->{gpu_fit} ? $file->{gpu_fit}{estimated_params} : 'N/A';
        say join(',',
            qq{"$file->{path}"},
            $file->{size_bytes},
            qq{"$file->{size_human}"},
            $file->{age_days},
            qq{"$file->{mtime_str}"},
            $gpu_status,
            qq{"$params"}
        );
    }
}
else {  # table format
    say "\n=== GGUF Model Files (GPU: $GPU_SPECS{name} - $GPU_SPECS{vram_gb}GB) ===\n";
    say sprintf("Total files found: %d\n", scalar(@sorted_files));

    # Calculate column widths
    my $max_path = 60;

    # Print header
    printf "%-*s  %10s  %8s  %19s",
        $max_path, "Path", "Size", "Age", "Modified";
    if ($config{show_gpu_fit}) {
        printf "  %8s  %10s", "GPU Fit", "Est.Params";
    }
    print "\n";
    print "-" x ($max_path + 50);
    print "-" x 25 if $config{show_gpu_fit};
    print "\n";

    # Print files
    for my $file (@sorted_files) {
        my $path = $file->{path};
        $path = substr($path, 0, $max_path - 3) . "..." if length($path) > $max_path;

        printf "%-*s  %10s  %3d days  %19s",
            $max_path,
            $path,
            $file->{size_human},
            $file->{age_days},
            $file->{mtime_str};

        if ($config{show_gpu_fit} && $file->{gpu_fit}) {
            my $fit = $file->{gpu_fit};
            my $status = $fit->{with_context} ? "Yes" :
                        $fit->{direct} ? "Tight" : "No";
            printf "  %8s  %10s", $status, $fit->{estimated_params};
        }

        print "\n";
    }

    # Summary statistics
    print "\n";
    say "=== Summary ===";
    my $total_size = 0;
    my $gpu_compatible = 0;

    for my $file (@sorted_files) {
        $total_size += $file->{size_bytes};
        $gpu_compatible++ if $file->{gpu_fit} && $file->{gpu_fit}{with_context};
    }

    say "Total size: " . human_readable_size($total_size);
    say "GPU compatible (with context): $gpu_compatible/" . scalar(@sorted_files) if $config{show_gpu_fit};
    say "Average size: " . human_readable_size($total_size / @sorted_files);

    # Suggest filenames for completion
    print "\n=== Model Names for Quick Reference ===\n";
    my %seen;
    for my $file (@sorted_files) {
        my ($name) = $file->{path} =~ /([^\/]+)\.gguf$/i;
        next if $seen{$name}++;
        say "  $name";
    }
}

__END__

=head1 NAME

find_gguf_files.pl - GGUF Model File Finder with GPU Memory Awareness

=head1 SYNOPSIS

    find_gguf_files.pl [OPTIONS]

    # Find models that fit in GPU
    find_gguf_files.pl --max-size=14GB --format=table

    # Export to JSON
    find_gguf_files.pl --dir=/mnt/models --format=json

=head1 DESCRIPTION

This script recursively searches for GGUF model files and analyzes their
compatibility with NVIDIA RTX 4080 (16GB VRAM). It provides size analysis,
age tracking, and GPU memory fit estimation.

=head1 FEATURES

=over 4

=item * Recursive directory scanning

=item * GPU memory compatibility analysis

=item * Multiple output formats (table, JSON, CSV)

=item * Size and age filtering

=item * Parameter count estimation

=back

=head1 AUTHOR

GRYPHGEN Project - Multi-Model LLM Infrastructure

=head1 LICENSE

MIT License

=cut

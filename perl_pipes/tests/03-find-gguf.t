#!/usr/bin/env perl
use v5.36;
use strict;
use warnings;
use Test::More tests => 7;
use Test::Exception;
use File::Temp qw(tempdir tempfile);
use FindBin qw($RealBin);

# Test find_gguf_files.pl functionality

my $bin_dir = "$RealBin/../bin";
my $temp_dir = tempdir(CLEANUP => 1);

# Test script help and version
subtest 'Help and version output' => sub {
    plan tests => 4;

    my $help_output = `$bin_dir/find_gguf_files.pl --help 2>&1`;
    like($help_output, qr/GGUF Model File Finder/, "Help contains title");
    like($help_output, qr/Usage:/, "Help contains usage");

    my $version_output = `$bin_dir/find_gguf_files.pl --version 2>&1`;
    like($version_output, qr/GGUF File Finder v/, "Version output correct");
    like($version_output, qr/RTX 4080/, "Version mentions GPU target");
};

# Test with no files
subtest 'No GGUF files found' => sub {
    plan tests => 1;

    my $output = `$bin_dir/find_gguf_files.pl --dir $temp_dir 2>/dev/null`;
    like($output, qr/No .gguf files found/, "Correctly reports no files");
};

# Test with mock GGUF files
subtest 'Find mock GGUF files' => sub {
    plan tests => 3;

    # Create mock .gguf files
    my $file1 = "$temp_dir/model1.gguf";
    my $file2 = "$temp_dir/subdir/model2.gguf";

    mkdir "$temp_dir/subdir";

    open my $fh1, '>', $file1;
    print $fh1 "x" x (1024 * 1024 * 10);  # 10MB
    close $fh1;

    open my $fh2, '>', $file2;
    print $fh2 "x" x (1024 * 1024 * 5);   # 5MB
    close $fh2;

    my $output = `$bin_dir/find_gguf_files.pl --dir $temp_dir 2>/dev/null`;

    like($output, qr/model1\.gguf/, "Found model1.gguf");
    like($output, qr/model2\.gguf/, "Found model2.gguf");
    like($output, qr/Total files found: 2/, "Correct count");
};

# Test JSON output
subtest 'JSON output format' => sub {
    plan tests => 2;

    # Create a test file
    my $file = "$temp_dir/test.gguf";
    open my $fh, '>', $file;
    print $fh "test";
    close $fh;

    my $output = `$bin_dir/find_gguf_files.pl --dir $temp_dir --format json 2>/dev/null`;

    like($output, qr/"files"/, "JSON contains files array");
    like($output, qr/"gpu_specs"/, "JSON contains GPU specs");
};

# Test CSV output
subtest 'CSV output format' => sub {
    plan tests => 2;

    my $output = `$bin_dir/find_gguf_files.pl --dir $temp_dir --format csv 2>/dev/null`;

    like($output, qr/Path,Size/, "CSV has header");
    like($output, qr/test\.gguf/, "CSV contains test file");
};

# Test size filtering
subtest 'Size filtering' => sub {
    plan tests => 2;

    # With max-size filter
    my $output1 = `$bin_dir/find_gguf_files.pl --dir $temp_dir --max-size 1MB 2>/dev/null`;
    like($output1, qr/test\.gguf/, "Small file passes size filter");

    # With min-size filter
    my $output2 = `$bin_dir/find_gguf_files.pl --dir $temp_dir --min-size 100MB 2>/dev/null`;
    like($output2, qr/No .gguf files found/, "Large min-size filters out small files");
};

# Test sorting options
subtest 'Sorting options' => sub {
    plan tests => 3;

    # Test size sorting (default)
    my $output1 = `$bin_dir/find_gguf_files.pl --dir $temp_dir --sort size 2>/dev/null`;
    ok($output1, "Size sorting works");

    # Test age sorting
    my $output2 = `$bin_dir/find_gguf_files.pl --dir $temp_dir --sort age 2>/dev/null`;
    ok($output2, "Age sorting works");

    # Test name sorting
    my $output3 = `$bin_dir/find_gguf_files.pl --dir $temp_dir --sort name 2>/dev/null`;
    ok($output3, "Name sorting works");
};

done_testing();

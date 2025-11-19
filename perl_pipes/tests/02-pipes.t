#!/usr/bin/env perl
use v5.36;
use strict;
use warnings;
use Test::More tests => 8;
use Test::Exception;
use File::Temp qw(tempdir);
use FindBin qw($RealBin);

# Test named pipe creation and cleanup

my $bin_dir = "$RealBin/../bin";
my $temp_dir = tempdir(CLEANUP => 1);

# Test pipe setup script
subtest 'Pipe setup script basics' => sub {
    plan tests => 3;

    # Test help output
    my $help_output = `$bin_dir/setup_pipes.sh --help 2>&1`;
    like($help_output, qr/Setup Named Pipes/, "Help message contains title");
    like($help_output, qr/Usage:/, "Help message contains usage");

    # Test version output
    my $version_output = `$bin_dir/setup_pipes.sh --version 2>&1`;
    like($version_output, qr/Setup Pipes v/, "Version output correct");
};

# Test pipe creation
subtest 'Create pipes in temp directory' => sub {
    plan tests => 3;

    # Create pipes in temp directory
    my $result = system("$bin_dir/setup_pipes.sh --dir $temp_dir --num 2 >/dev/null 2>&1");
    is($result, 0, "Pipe setup completed successfully");

    # Verify pipes were created
    ok(-p "$temp_dir/model1_to_model2", "model1_to_model2 pipe created");
    ok(-p "$temp_dir/model2_to_model1", "model2_to_model1 pipe created");
};

# Test pipe cleanup
subtest 'Cleanup pipes' => sub {
    plan tests => 3;

    # Cleanup pipes
    my $result = system("$bin_dir/cleanup_pipes.sh --dir $temp_dir --force >/dev/null 2>&1");
    is($result, 0, "Pipe cleanup completed successfully");

    # Verify pipes were removed
    ok(!-p "$temp_dir/model1_to_model2", "model1_to_model2 pipe removed");
    ok(!-p "$temp_dir/model2_to_model1", "model2_to_model1 pipe removed");
};

# Test multiple models
subtest 'Create pipes for multiple models' => sub {
    plan tests => 7;

    # Create pipes for 3 models
    my $result = system("$bin_dir/setup_pipes.sh --dir $temp_dir --num 3 >/dev/null 2>&1");
    is($result, 0, "Pipe setup for 3 models completed");

    # Verify all 6 pipes were created (3 models = 6 bidirectional pipes)
    ok(-p "$temp_dir/model1_to_model2", "model1_to_model2 exists");
    ok(-p "$temp_dir/model2_to_model1", "model2_to_model1 exists");
    ok(-p "$temp_dir/model1_to_model3", "model1_to_model3 exists");
    ok(-p "$temp_dir/model3_to_model1", "model3_to_model1 exists");
    ok(-p "$temp_dir/model2_to_model3", "model2_to_model3 exists");
    ok(-p "$temp_dir/model3_to_model2", "model3_to_model2 exists");

    # Cleanup
    system("$bin_dir/cleanup_pipes.sh --dir $temp_dir --force >/dev/null 2>&1");
};

# Test custom prefix
subtest 'Create pipes with custom prefix' => sub {
    plan tests => 3;

    my $result = system("$bin_dir/setup_pipes.sh --dir $temp_dir --prefix llm --num 2 >/dev/null 2>&1");
    is($result, 0, "Pipe setup with custom prefix completed");

    ok(-p "$temp_dir/llm1_to_llm2", "llm1_to_llm2 pipe created");
    ok(-p "$temp_dir/llm2_to_llm1", "llm2_to_llm1 pipe created");

    # Cleanup
    system("$bin_dir/cleanup_pipes.sh --dir $temp_dir --prefix llm --force >/dev/null 2>&1");
};

# Test pipe recreation (clean option)
subtest 'Recreate existing pipes with --clean' => sub {
    plan tests => 4;

    # Create initial pipes
    system("$bin_dir/setup_pipes.sh --dir $temp_dir --num 2 >/dev/null 2>&1");

    # Get initial inode
    my @stat1 = stat("$temp_dir/model1_to_model2");
    my $inode1 = $stat1[1];

    # Recreate with --clean
    my $result = system("$bin_dir/setup_pipes.sh --dir $temp_dir --num 2 --clean >/dev/null 2>&1");
    is($result, 0, "Pipe recreation completed");

    # Get new inode
    my @stat2 = stat("$temp_dir/model1_to_model2");
    my $inode2 = $stat2[1];

    # Pipes should exist
    ok(-p "$temp_dir/model1_to_model2", "Pipe still exists after recreation");
    ok(-p "$temp_dir/model2_to_model1", "Pipe still exists after recreation");

    # Inode should be different (new pipe)
    isnt($inode1, $inode2, "Pipe was recreated (different inode)");

    # Cleanup
    system("$bin_dir/cleanup_pipes.sh --dir $temp_dir --force >/dev/null 2>&1");
};

# Test error handling
subtest 'Error handling for invalid directory' => sub {
    plan tests => 1;

    my $result = system("$bin_dir/setup_pipes.sh --dir /nonexistent/directory 2>/dev/null");
    isnt($result, 0, "Setup fails for nonexistent directory");
};

# Test cleanup help and version
subtest 'Cleanup script help and version' => sub {
    plan tests => 2;

    my $help_output = `$bin_dir/cleanup_pipes.sh --help 2>&1`;
    like($help_output, qr/Cleanup Named Pipes/, "Cleanup help message correct");

    my $version_output = `$bin_dir/cleanup_pipes.sh --version 2>&1`;
    like($version_output, qr/Cleanup Pipes v/, "Cleanup version output correct");
};

done_testing();

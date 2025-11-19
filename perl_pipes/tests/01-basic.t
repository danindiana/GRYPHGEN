#!/usr/bin/env perl
use v5.36;
use strict;
use warnings;
use Test::More tests => 12;
use Test::Exception;
use File::Temp qw(tempdir);
use FindBin qw($RealBin);

# Test basic script existence and permissions

my $bin_dir = "$RealBin/../bin";

# Test 1-5: Check script files exist
ok(-f "$bin_dir/model1_comm.pl", "model1_comm.pl exists");
ok(-f "$bin_dir/model2_comm.pl", "model2_comm.pl exists");
ok(-f "$bin_dir/find_gguf_files.pl", "find_gguf_files.pl exists");
ok(-f "$bin_dir/setup_pipes.sh", "setup_pipes.sh exists");
ok(-f "$bin_dir/cleanup_pipes.sh", "cleanup_pipes.sh exists");

# Test 6-10: Check scripts are executable
ok(-x "$bin_dir/model1_comm.pl", "model1_comm.pl is executable");
ok(-x "$bin_dir/model2_comm.pl", "model2_comm.pl is executable");
ok(-x "$bin_dir/find_gguf_files.pl", "find_gguf_files.pl is executable");
ok(-x "$bin_dir/setup_pipes.sh", "setup_pipes.sh is executable");
ok(-x "$bin_dir/cleanup_pipes.sh", "cleanup_pipes.sh is executable");

# Test 11: Check scripts have valid Perl syntax
lives_ok {
    system("$^X -c $bin_dir/model1_comm.pl 2>/dev/null") == 0
        or die "model1_comm.pl has syntax errors";
} "model1_comm.pl has valid syntax";

# Test 12: Check find_gguf_files has valid syntax
lives_ok {
    system("$^X -c $bin_dir/find_gguf_files.pl 2>/dev/null") == 0
        or die "find_gguf_files.pl has syntax errors";
} "find_gguf_files.pl has valid syntax";

done_testing();

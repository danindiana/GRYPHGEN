Here's how you can use Perl instead of Python to interact with the LLM and execute code:

1. **Install necessary libraries**: You'll need `Ollama::Client` for interacting with Ollama and `Inline::Perl5` for executing Perl code. Install them using cpan:

   ```
   cpan install Ollama::Client Inline::Perl5
   ```

2. **Create a Modelfile** (without extension) for your GGUF model, specifying the path to your model file and any additional configurations.

3. **Link your Modelfile** to your GGUF model using the following command:

   ```
   ollama create <model-name> -f /path/to/your/Modelfile
   ```

4. **Create a Perl script (e.g., `main.pl`)** with the following content, replacing `<model-name>` and `/path/to/Modelfile` accordingly.

```perl
#!/usr/bin/perl

use Ollama::Client;
use Inline Perl5;
use strict;
use warnings;

my $client = Ollama::Client->new();
my $history = [{ role => 'system', content => `cat /path/to/your/Modelfile` }];
print "OLLAMA Chat Interface. Press Ctrl+C to interrupt the response or Ctrl+D to exit.\n";

while (1) {
    print "> ";
    my $user_input = <STDIN>;
    chomp($user_input);

    if ($user_input eq "/exit") {
        print "Exiting chat.\n";
        last;
    }

    push @$history, { role => 'user', content => $user_input };

    try {
        my $response = $client->chat(model => '<model-name>', messages => $history, stream => 1);
        my $full_response = '';
        while (my $chunk = $response->read) {
            if ($chunk->{message}->{content}) {
                print $chunk->{message}->{content};
                $full_response .= $chunk->{message}->{content};

                # Check for Perl code block
                if ($full_response =~ m/```Perl\n(.*?)```/) {
                    my $perl_code = $1;
                    my $execution_result = execute_Perl_code($perl_code);
                    print "\nExecuted Result: $execution_result\n";
                    push @$history, { role => 'assistant', content => $full_response };
                    push @$history, { role => 'user', content => "Executed Result: $execution_result" };
                }
            }
        }
    } catch {
        print "Error: $_\n";
    }

    push @$history, { role => 'assistant', content => $full_response };
}

sub execute_Perl_code {
    my ($perl_code) = @_;
    return eval_perl($perl_code);
}
```

5. **Run your Perl script**:

   ```
   perl main.pl
   ```

Now, you can interact with the LLM using Perl, and it will execute any Perl code enclosed in triple backticks (````Perl\ncode here\n````) as specified in the system message.

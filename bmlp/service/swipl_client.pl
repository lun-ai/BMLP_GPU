:- use_module(library(socket)).

run_python_command(Command, Result) :-
    tcp_socket(Socket),
    tcp_connect(Socket, '127.0.0.1':8888),
    setup_call_cleanup(
        tcp_open_socket(Socket, In, Out),
        (
            format(Out, '~w~n', [Command]),
            flush_output(Out),
            read_string(In, _, Response),
            string_codes(Result, Response)
        ),
        close_connection(In, Out)
    ).

close_connection(In, Out) :-
    close(In, [force(true)]),
    close(Out, [force(true)]).

% 示例调用

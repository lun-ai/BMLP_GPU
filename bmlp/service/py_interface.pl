:- use_module(library(janus)).

% If using Janus to call python code, need to first configure the environment
update_env(EnvPath) :-
    py_version,
    py_call(sys:path, Paths),
    forall(
        (member(Path, Paths),
        (sub_string(Path, _, _, 0, 'dist-packages') ; sub_string(Path, _, _, 0, 'site-packages'))),
        py_call(sys:path:remove(Path))
        ),
    py_call(sys:path:append(EnvPath)),
    writeln('% Use updated Python Virtual Enviroment'),!.
update_env(_) :-
    throw(existence_error(virtual_env_not_found,_)).
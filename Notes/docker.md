### Useful docker commands

- Run a container with /bin/bash as the shell

`> docker run -i -t ubuntu /bin/bash`

---

- Docker uses a technology called namespaces to provide the isolated workspace called the container.
---

- Run `getting-started` container in detached mode with port 80 of the host mapped to port 80 of the container.

`> docker run -d -p 80:80 docker/getting-started`

---

- see running containers:

`> docker container ls`

---

- see a list of images present at system:

`> docker images`

---

- to build an image from dockerfile:

`> docker build --tag directgrav .`

---

- see running dockers:

`> docker ps`

to see all of them including exited dockers, supply with `-a`.

---

- to start an exited docker:

`> docker start DOKCER-NAME`

- to stop it:

`> docker stop DOCKER-NAME-OR-ID`

---

- to remove a docker from processes:

`> docker rm DOCKER-NAME`

this still will have the image, it just removes from kind of at sleep dockers.

---

- to follow logs of a container, running in detached mode:

`> docker logs -f DOCKER-NAME`

---

- to tag a local image:

`> docker tag directgrav saliei/directgrav`

---

- to push a local image to docker hub, first login to docker:

`> docker login`

and then:

`> docker push saliei/directgrav`

---

- to remove an image:

`> docker rmi IMAGE-NAME`

--- 

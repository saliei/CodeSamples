import express from 'express'
import os from 'os'

const app = express()
const PORT = 3000

app.get("/", (req, res) => {
    const msg = `V2.0: I am host: ${os.hostname()}`
    console.log(msg)
    res.send(msg)
})

app.listen(PORT, () => {
    console.log(`web server listening on port: ${PORT}`)
})

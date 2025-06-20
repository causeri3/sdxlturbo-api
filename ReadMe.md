# Stable Diffusion XL Turbo API
For an art installation, [this](https://github.com/causeri3/selfusion-pi) is the main repo.

___

### Deployment
It's deployed with Runpod and was impressively easy in contrast to your typical big clouds provider. 
I straight used Runpods image as base in my Dockerfile, knowing it works.

#### local
```bash
docker buildx build --platform linux/amd64 -t yourusername/sdxlturbo-api:latest .
docker login
docker push yourusername/sdxlturbo-api:latest
```
or just the one I pushed:

`causeri3/sdxlturbo-api:latest`
#### Runpod UI
- choose the default RTX3090 Pod
- in the settings: 
  - put in the Docker image 
  - expose port 8000
- *sweet*: once its running, you can click on connect - it straight gives you a public url
- *optional*: in your profile you can upload your public ssh key and it will automatically assign it to your pod
___

### Performance
In case it helps anyone to make a decision, for comparison, the current setup:

| Seconds | GPU    | RAM    | GPU computing frameworks |
|---------|--------|--------|--------------------------|
| 24      | RTX3090 | 125 GB | cuda                     |
| 210     | M3 Pro | 36 GB  | mps                      |
___
*Powered by Stability AI*

This API uses the Stability AI model [SDXL Turbo](https://huggingface.co/stabilityai/sdxl-turbo) under the [Stability AI Community License](https://huggingface.co/stabilityai/sdxl-turbo/blob/main/LICENSE.md).

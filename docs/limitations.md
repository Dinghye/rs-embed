## Todo

### Urgent
- [ ]  Check the meta data format, unify name
- [ ] Add api document
- [ ] **important**: efficiency issue in batch acquisition
    - [x] add batch function in EmbedderBase, for better batch process(overwrite in different model, default for loop)
    - [x] Repeated downloads / Repeated initializations!!!
- [x] interface check. ModelError
- [x] Remove print(), prevent from polluting stdout
- [x] Perform basic checks for spatial/temporal/output uniformly (to avoid each embedder reporting their own errors separately)-> api.py

### Future
- [ ] More backend? Mircosoft Planetary? Open Street Map?
- [ ] Load local image for embedding?
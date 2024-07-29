# Website

This website is built using [Docusaurus](https://docusaurus.io/), a modern static website generator.

### Prerequisites

- Install [volta](https://docs.volta.sh/guide/getting-started)
  - `volta install node`: install last version of `node`
  - `volta install yarn`: install last version of `yarn`

### Installation

```
$ yarn
```

### Local Development

```
$ yarn start
```

This command starts a local development server and opens up a browser window. Most changes are reflected live without having to restart the server.

### Update the documentation

- Update the markdown files in the `content` directory.

### Deployment

To deploy the website in our GitHub pages:

- run `yarn build` to generate the static content.
- Commit the changes and push to update the website.

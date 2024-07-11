"use strict";(self.webpackChunksnnax_docs=self.webpackChunksnnax_docs||[]).push([[871],{9848:(e,n,r)=>{r.r(n),r.d(n,{assets:()=>o,contentTitle:()=>i,default:()=>h,frontMatter:()=>c,metadata:()=>a,toc:()=>l});var t=r(4848),s=r(8453);const c={},i="Introduction",a={id:"architecture/intro",title:"Introduction",description:"\x3c!-- class GraphStructure(NamedTuple):",source:"@site/docs/200_architecture/200_intro.md",sourceDirName:"200_architecture",slug:"/architecture/intro",permalink:"/snnax/architecture/intro",draft:!1,unlisted:!1,editUrl:"https://iffgit.fz-juelich.de/pgi-15/snnax-docs/docs/200_architecture/200_intro.md",tags:[],version:"current",sidebarPosition:200,frontMatter:{},sidebar:"tutorialSidebar",previous:{title:"Structure",permalink:"/snnax/gettingStarted/structure"},next:{title:"Composed",permalink:"/snnax/architecture/composed"}},o={},l=[{value:"StatefulModel",id:"statefulmodel",level:2},{value:"GraphStructure",id:"graphstructure",level:3}];function d(e){const n={a:"a",code:"code",h1:"h1",h2:"h2",h3:"h3",li:"li",mermaid:"mermaid",p:"p",pre:"pre",ul:"ul",...(0,s.R)(),...e.components};return(0,t.jsxs)(t.Fragment,{children:[(0,t.jsx)(n.h1,{id:"introduction",children:"Introduction"}),"\n",(0,t.jsx)(n.p,{children:"In order to customize model's architecture, SNNAX provides a simple way to define the architecture of the model."}),"\n",(0,t.jsx)(n.p,{children:"SNNAX provides two ways to define the architecture of the model:"}),"\n",(0,t.jsxs)(n.ul,{children:["\n",(0,t.jsxs)(n.li,{children:[(0,t.jsx)(n.a,{href:"/snnax/architecture/intro#statefulmodel",children:(0,t.jsx)(n.code,{children:"snnax.snn.architecture.StatefulModel"})}),": to create custom SNNs."]}),"\n",(0,t.jsxs)(n.li,{children:[(0,t.jsx)(n.a,{href:"/snnax/architecture/composed",children:(0,t.jsx)(n.code,{children:"snnax.snn.composed"})}),": provides classes of predefined architectures that can be used to build your model."]}),"\n"]}),"\n",(0,t.jsx)(n.h2,{id:"statefulmodel",children:"StatefulModel"}),"\n",(0,t.jsxs)(n.p,{children:["The ",(0,t.jsx)(n.code,{children:"StatefulModel"})," class allows the creation of custom SNNs with almost arbitrary connectivity defined through a graph structure called the connectivity graph. It has to inherit from ",(0,t.jsx)(n.code,{children:"eqx.Module"})," to be a callable pytree."]}),"\n",(0,t.jsx)(n.p,{children:"It requires the following arguments:"}),"\n",(0,t.jsxs)(n.ul,{children:["\n",(0,t.jsxs)(n.li,{children:[(0,t.jsx)(n.code,{children:"graph_structure (GraphStructure)"}),": GraphStructure object to specify network topology."]}),"\n",(0,t.jsxs)(n.li,{children:[(0,t.jsx)(n.code,{children:"layers (Sequence[eqx.Module])"}),": Computational building blocks of the model."]}),"\n",(0,t.jsxs)(n.li,{children:[(0,t.jsx)(n.code,{children:"forward_fn (Callable)"}),": Evaluation procedure/loop for the model. Defaults to ",(0,t.jsx)(n.a,{href:"/snnax/functions/intro",children:(0,t.jsx)(n.code,{children:"forward_fn"})}),"."]}),"\n"]}),"\n",(0,t.jsxs)(n.p,{children:["First we need to define our layers that can be from the ",(0,t.jsx)(n.code,{children:"equinox"})," library or SNNAX ",(0,t.jsx)(n.a,{href:"/snnax/layers/intro",children:"snnax.snn.layers"}),"."]}),"\n",(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{className:"language-python",children:"from snnax import snn\nimport equinox as eqx\n\nlayers = [eqx.Linear(),\n           eqx.LayerNorm(),\n           snn.LIF()]\n"})}),"\n",(0,t.jsxs)(n.p,{children:["Then we need to define the ",(0,t.jsx)(n.a,{href:"/snnax/architecture/intro#graphstructure",children:(0,t.jsx)(n.code,{children:"GraphStructure"})})," object which contains meta-information about the computational graph."]}),"\n",(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{className:"language-python",children:"graph_structure = snn.GraphStructure(3, [[0], [], [],], [2], [[2], [0], [1]])\n"})}),"\n",(0,t.jsxs)(n.p,{children:["Finally, we can create the ",(0,t.jsx)(n.code,{children:"StatefulModel"})," object by passing the ",(0,t.jsx)(n.code,{children:"graph_structure"})," and ",(0,t.jsx)(n.code,{children:"layers"})," as arguments."]}),"\n",(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{className:"language-python",children:"model = snn.StatefulModel(graph_structure=graph_structure,\n                            layers=layers)\n"})}),"\n",(0,t.jsx)(n.p,{children:"and the model architecture will be:"}),"\n",(0,t.jsx)(n.mermaid,{value:"graph LR;\n    Input--\x3eLinear;\n    Linear--\x3eLayerNorm;\n    LayerNorm--\x3eLIF;\n    LIF--\x3eLinear;\n    LIF--\x3eOutput;"}),"\n",(0,t.jsx)(n.h3,{id:"graphstructure",children:"GraphStructure"}),"\n",(0,t.jsxs)(n.p,{children:["The ",(0,t.jsx)(n.code,{children:"GraphStructure"})," class contains meta-information about the computational graph. It can be used in conjunction with the ",(0,t.jsx)(n.code,{children:"StatefulModel"})," class to construct a computational model."]}),"\n",(0,t.jsxs)(n.p,{children:["The ",(0,t.jsx)(n.code,{children:"GraphStructure"})," class requires the following arguments:"]}),"\n",(0,t.jsxs)(n.ul,{children:["\n",(0,t.jsxs)(n.li,{children:[(0,t.jsx)(n.code,{children:"num_layers (int)"}),": The number of layers we want to have in our model."]}),"\n",(0,t.jsxs)(n.li,{children:[(0,t.jsx)(n.code,{children:"input_layer_ids (Sequence[Sequence[int]])"}),": Index of the layers are provided with external input."]}),"\n",(0,t.jsxs)(n.li,{children:[(0,t.jsx)(n.code,{children:"final_layer_ids (Sequence[int])"}),": Index of the layers whose output we want."]}),"\n",(0,t.jsxs)(n.li,{children:[(0,t.jsx)(n.code,{children:"input_connectivity (Sequence[Sequence[int]])"}),": Specifies how the layers are connected to each other."]}),"\n"]}),"\n",(0,t.jsx)(n.p,{children:"Example:"}),"\n",(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{className:"language-python",children:"graph_structure = snn.GraphStructure(6, [[0], [], [], [3], [], []], [5], [[], [0, 2], [1], [2, 3], [3], [4]])\n"})}),"\n",(0,t.jsx)(n.mermaid,{value:"graph LR;\n    Input--\x3e0;\n    Input--\x3e3;\n    0--\x3e1;\n    2--\x3e1;\n    1--\x3e2;\n    2--\x3e3;\n    3--\x3e4;\n    4--\x3e4;\n    4--\x3e5;\n    5--\x3eOutput;"})]})}function h(e={}){const{wrapper:n}={...(0,s.R)(),...e.components};return n?(0,t.jsx)(n,{...e,children:(0,t.jsx)(d,{...e})}):d(e)}}}]);
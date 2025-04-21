/// <reference types="vite/client" />

// 声明 Vite 导入方式，允许导入各种资源
declare module '*.svg' {
  import React = require('react');
  export const ReactComponent: React.FunctionComponent<React.SVGProps<SVGSVGElement>>;
  const src: string;
  export default src;
}

declare module '*.jpg' {
  const src: string;
  export default src;
}

declare module '*.jpeg' {
  const src: string;
  export default src;
}

declare module '*.png' {
  const src: string;
  export default src;
}

// 声明 @vitejs/plugin-react 模块，让 TypeScript 不再报错
declare module '@vitejs/plugin-react' {
  import { Plugin } from 'vite';
  function react(options?: any): Plugin;
  export default react;
}

// 如果你看到其他模块的类型错误，也可以在这里声明 
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
// import path from 'path' // path is a Node.js built-in, @types/node provides types
import { fileURLToPath } from 'node:url'

// const shimPath = fileURLToPath(new URL('./src/uuid-shim.js', import.meta.url)); // Old way
// const topLevelShimPath = '/app/src/uuid-shim.js'; // No longer needed
// console.log(`vite.config.ts: Top-level shimPath defined as: ${topLevelShimPath}`); // No longer needed

// const __filename = fileURLToPath(import.meta.url);
// const __dirname = path.dirname(__filename);

// https://vitejs.dev/config/
export default defineConfig({
  // logLevel: 'debug', // Removed due to type error, Vite's dev logging is usually verbose enough
  define: {
    global: 'window',
  },
  plugins: [react()],
  server: {
    host: '0.0.0.0', // 允许外部访问
    port: 5173,      // 开发服务器端口
    proxy: {
      // 将 /api 开头的请求代理到后端容器
      // 'app' 是 compose.yml 中后端服务的名称
      '/api': {
        target: 'http://app:8000', // 修改为后端服务的容器名和端口
        changeOrigin: true,       // 需要虚拟主机站点
        // secure: false,         // 如果后端是 https 但证书无效，可能需要
        // rewrite: (path) => path.replace(/^\/api/, ''), // 如果后端路径不包含 /api，取消注释此行
      },
    },
    // 解决 Podman + Vite HMR 问题
    // watch 选项应放在 server 内部
    watch: {
      usePolling: true,
    },
  },
  resolve: {
    // alias: [ // UUID ALIAS NO LONGER NEEDED
    //   {
    //     find: 'uuid',
    //     replacement: (
    //       source: string, 
    //       importer: string | undefined, 
    //       options: { isBuild: boolean; customResolverOptions?: any }
    //     ): string | undefined => {
    //       console.log(`vite.config.ts: Custom alias for 'uuid'. Source: '${source}', Importer: '${importer}', isBuild: ${options.isBuild}`);
    //       if (importer && importer.includes('uuid-shim.js')) {
    //         console.log(`vite.config.ts: 'uuid' import from shim itself ('${importer}'). Letting Vite resolve original 'uuid' from node_modules.`);
    //         return undefined; // Let Vite resolve 'uuid' from node_modules normally
    //       }
    //       console.log(`vite.config.ts: 'uuid' import from '${importer}'. Using shim: '${topLevelShimPath}'`);
    //       return topLevelShimPath;
    //     },
    //   } as any // Temporarily cast to any to bypass TypeScript error and test runtime behavior
    // ]
  },
  optimizeDeps: {
    include: [
      'react-graph-vis',
      // 'uuid' // No longer explicitly needed here due to override and patch
    ],
    // exclude: [ // <--- Remove or comment out exclude for now
    //   'uuid'
    // ],
    esbuildOptions: {
      loader: {
        '.js': 'jsx'
      },
      // If uuid-shim.js is pure CommonJS, a CJS plugin for esbuild might be needed here
      // For now, assuming esbuild handles .js with module.exports correctly with the loader.
      // plugins: [
      //   // esbuild CJS plugin if necessary
      // ]
    }
  }
}) 